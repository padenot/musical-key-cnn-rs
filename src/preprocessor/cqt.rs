// Derived from rosa 0.1.0's MIT-licensed librosa-compatible CQT implementation.
// See LICENSE-ROSA. This module keeps only the fixed MusicalKeyCNN transform.

use std::borrow::Cow;
use std::error::Error;
use std::fmt;

use realfft::RealFftPlanner;

use super::resample::resample;
use super::{BINS_PER_OCTAVE, FREQUENCY_BINS, HOP_LENGTH, MINIMUM_FREQUENCY_HZ, MODEL_SAMPLE_RATE};

pub(super) struct CqtWorkspace {
    fft_planner: RealFftPlanner<f64>,
}

impl CqtWorkspace {
    pub(super) fn new() -> Self {
        Self {
            fft_planner: RealFftPlanner::new(),
        }
    }
}

#[derive(Debug)]
pub(super) enum CqtError {
    InvalidParameters(String),
    Resample(String),
    Fft(String),
}

impl fmt::Display for CqtError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidParameters(message) => write!(formatter, "invalid parameters: {message}"),
            Self::Resample(message) => write!(formatter, "resampling failed: {message}"),
            Self::Fft(message) => write!(formatter, "FFT failed: {message}"),
        }
    }
}

impl Error for CqtError {}

pub(super) struct Magnitude {
    data: Vec<f64>,
    rows: usize,
    columns: usize,
}

impl Magnitude {
    fn zeros(rows: usize, columns: usize) -> Result<Self, CqtError> {
        let length = rows
            .checked_mul(columns)
            .ok_or_else(|| CqtError::InvalidParameters("CQT matrix size overflow".to_owned()))?;
        Ok(Self {
            data: vec![0.0; length],
            rows,
            columns,
        })
    }

    fn row(&self, row: usize) -> Option<&[f64]> {
        let start = row.checked_mul(self.columns)?;
        let end = start.checked_add(self.columns)?;
        self.data.get(start..end)
    }

    fn row_mut(&mut self, row: usize) -> Option<&mut [f64]> {
        let start = row.checked_mul(self.columns)?;
        let end = start.checked_add(self.columns)?;
        self.data.get_mut(start..end)
    }

    pub(super) fn rows(&self) -> usize {
        self.rows
    }

    pub(super) fn cols(&self) -> usize {
        self.columns
    }

    pub(super) fn shape(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    pub(super) fn as_slice(&self) -> &[f64] {
        &self.data
    }

    pub(super) fn map(&self, transform: impl Fn(f64) -> f64) -> Self {
        Self {
            data: self.data.iter().copied().map(transform).collect(),
            rows: self.rows,
            columns: self.columns,
        }
    }
}

struct FilterBasis {
    real: Magnitude,
    imaginary: Magnitude,
    fft_size: usize,
}

pub(super) fn cqt_magnitude(
    samples: &[f64],
    workspace: &mut CqtWorkspace,
) -> Result<Magnitude, CqtError> {
    validate_samples(samples)?;
    let octave_count = FREQUENCY_BINS.div_ceil(BINS_PER_OCTAVE);
    let frequencies = cqt_frequencies();
    let (filter_lengths, cutoff_frequency) = wavelet_lengths(&frequencies)?;
    let sample_rate = f64::from(MODEL_SAMPLE_RATE);
    let nyquist = sample_rate / 2.0;
    let divisible_twos = HOP_LENGTH.trailing_zeros() as usize;
    let downsample_count = if cutoff_frequency > 0.0 && nyquist > cutoff_frequency {
        let available = (nyquist / cutoff_frequency).log2().floor() as isize;
        let octave_offset = isize::try_from(octave_count.saturating_sub(1)).map_err(|source| {
            CqtError::InvalidParameters(format!("octave count cannot be represented: {source}"))
        })?;
        usize::try_from((available - 1 - octave_offset).max(0))
            .map_or(0, |count| count.min(divisible_twos))
    } else {
        0
    };

    let mut signal = if downsample_count > 0 {
        let factor = 1_u32
            .checked_shl(downsample_count as u32)
            .ok_or_else(|| CqtError::InvalidParameters("downsample factor overflow".to_owned()))?;
        let output_rate = MODEL_SAMPLE_RATE
            .checked_div(factor)
            .ok_or_else(|| CqtError::InvalidParameters("downsample rate is zero".to_owned()))?;
        let mut downsampled =
            resample(samples, MODEL_SAMPLE_RATE, output_rate).map_err(CqtError::Resample)?;
        let energy_scale = f64::from(factor).sqrt();
        downsampled
            .iter_mut()
            .for_each(|sample| *sample *= energy_scale);
        Cow::Owned(downsampled)
    } else {
        Cow::Borrowed(samples)
    };
    let mut current_rate = sample_rate / 2.0_f64.powi(downsample_count as i32);
    let mut hop_length = HOP_LENGTH >> downsample_count;
    let mut responses = Vec::with_capacity(octave_count);

    for octave in 0..octave_count {
        let bin_start =
            FREQUENCY_BINS.saturating_sub(octave.saturating_add(1).saturating_mul(BINS_PER_OCTAVE));
        let bin_end = FREQUENCY_BINS.saturating_sub(octave.saturating_mul(BINS_PER_OCTAVE));
        let octave_bins = bin_end.saturating_sub(bin_start);
        if octave_bins == 0 {
            continue;
        }
        let octave_frequencies = frequencies.get(bin_start..bin_end).ok_or_else(|| {
            CqtError::InvalidParameters("octave frequency range is invalid".to_owned())
        })?;
        let mut basis = filter_basis(current_rate, octave_frequencies)?;
        let rate_scale = (sample_rate / current_rate).sqrt();
        basis
            .real
            .data
            .iter_mut()
            .for_each(|value| *value *= rate_scale);
        basis
            .imaginary
            .data
            .iter_mut()
            .for_each(|value| *value *= rate_scale);
        responses.push((
            magnitude_response(signal.as_ref(), hop_length, &basis, workspace)?,
            octave_bins,
        ));

        if octave < octave_count.saturating_sub(1) && hop_length.is_multiple_of(2) {
            let input_rate = current_rate as u32;
            let output_rate = input_rate / 2;
            if output_rate > 0 {
                let mut downsampled = resample(signal.as_ref(), input_rate, output_rate)
                    .map_err(CqtError::Resample)?;
                downsampled
                    .iter_mut()
                    .for_each(|sample| *sample *= std::f64::consts::SQRT_2);
                signal = Cow::Owned(downsampled);
                current_rate /= 2.0;
                hop_length /= 2;
            }
        }
    }

    let mut magnitude = stack_octaves(&responses)?;
    for (bin, length) in filter_lengths.iter().copied().enumerate() {
        let row = magnitude
            .row_mut(bin)
            .ok_or_else(|| CqtError::InvalidParameters("CQT output row is missing".to_owned()))?;
        let scale = 1.0 / length.sqrt();
        row.iter_mut().for_each(|value| *value *= scale);
    }
    Ok(magnitude)
}

fn validate_samples(samples: &[f64]) -> Result<(), CqtError> {
    if samples.is_empty() {
        return Err(CqtError::InvalidParameters("input is empty".to_owned()));
    }
    if samples.iter().any(|sample| !sample.is_finite()) {
        return Err(CqtError::InvalidParameters(
            "input contains non-finite samples".to_owned(),
        ));
    }
    Ok(())
}

fn cqt_frequencies() -> Vec<f64> {
    (0..FREQUENCY_BINS)
        .map(|bin| MINIMUM_FREQUENCY_HZ * 2.0_f64.powf(bin as f64 / BINS_PER_OCTAVE as f64))
        .collect()
}

fn wavelet_lengths(frequencies: &[f64]) -> Result<(Vec<f64>, f64), CqtError> {
    if frequencies.len() < 2 {
        return Err(CqtError::InvalidParameters(
            "at least two CQT frequencies are required".to_owned(),
        ));
    }
    let ratio = 2.0_f64.powf(1.0 / BINS_PER_OCTAVE as f64);
    let relative_bandwidth = (ratio * ratio - 1.0) / (ratio * ratio + 1.0);
    let quality = 1.0 / relative_bandwidth;
    let sample_rate = f64::from(MODEL_SAMPLE_RATE);
    let lengths = frequencies
        .iter()
        .map(|frequency| quality * sample_rate / frequency)
        .collect::<Vec<_>>();
    let window_bandwidth = hann_window_bandwidth(1_000)?;
    let cutoff = frequencies
        .iter()
        .map(|frequency| frequency * (1.0 + 0.5 * window_bandwidth / quality))
        .fold(0.0_f64, f64::max);
    Ok((lengths, cutoff))
}

fn hann_window_bandwidth(length: usize) -> Result<f64, CqtError> {
    let window = hann(length);
    let sum = window.iter().sum::<f64>();
    if sum.abs() <= f64::MIN_POSITIVE {
        return Err(CqtError::InvalidParameters(
            "Hann window has zero energy".to_owned(),
        ));
    }
    Ok(length as f64 * window.iter().map(|value| value * value).sum::<f64>() / (sum * sum))
}

fn hann(length: usize) -> Vec<f64> {
    if length == 0 {
        return Vec::new();
    }
    if length == 1 {
        return vec![1.0];
    }
    let extended_length = length.saturating_add(1);
    (0..extended_length)
        .map(|index| {
            let phase = std::f64::consts::TAU * index as f64 / (extended_length - 1) as f64;
            0.5 - 0.5 * phase.cos()
        })
        .take(length)
        .collect()
}

fn filter_basis(sample_rate: f64, frequencies: &[f64]) -> Result<FilterBasis, CqtError> {
    let ratio = 2.0_f64.powf(1.0 / BINS_PER_OCTAVE as f64);
    let relative_bandwidth = (ratio * ratio - 1.0) / (ratio * ratio + 1.0);
    let quality = 1.0 / relative_bandwidth;
    let lengths = frequencies
        .iter()
        .map(|frequency| quality * sample_rate / frequency)
        .collect::<Vec<_>>();
    let mut filters = Vec::with_capacity(frequencies.len());

    for (&frequency, &length) in frequencies.iter().zip(&lengths) {
        let start = (-length / 2.0).floor() as i64;
        let end = (length / 2.0).floor() as i64;
        let filter_length = usize::try_from((end - start).max(0)).map_err(|source| {
            CqtError::InvalidParameters(format!("filter length cannot be represented: {source}"))
        })?;
        let window = hann(filter_length);
        let mut real = Vec::with_capacity(filter_length);
        let mut imaginary = Vec::with_capacity(filter_length);
        for (offset, window_value) in window.into_iter().enumerate() {
            let time = (start + offset as i64) as f64;
            let phase = std::f64::consts::TAU * frequency * time / sample_rate;
            real.push(phase.cos() * window_value);
            imaginary.push(phase.sin() * window_value);
        }
        let norm = real
            .iter()
            .zip(&imaginary)
            .map(|(&real, &imaginary)| real.hypot(imaginary))
            .sum::<f64>();
        if norm <= f64::MIN_POSITIVE {
            return Err(CqtError::InvalidParameters(
                "CQT filter has zero norm".to_owned(),
            ));
        }
        real.iter_mut().for_each(|value| *value /= norm);
        imaginary.iter_mut().for_each(|value| *value /= norm);
        filters.push((real, imaginary, length));
    }

    let longest = filters
        .iter()
        .map(|(real, _, _)| real.len())
        .max()
        .ok_or_else(|| CqtError::InvalidParameters("CQT filter bank is empty".to_owned()))?;
    let fft_size = longest
        .checked_next_power_of_two()
        .ok_or_else(|| CqtError::InvalidParameters("CQT FFT size overflow".to_owned()))?;
    let spectrum_bins = fft_size / 2 + 1;
    let mut real_basis = Magnitude::zeros(frequencies.len(), spectrum_bins)?;
    let mut imaginary_basis = Magnitude::zeros(frequencies.len(), spectrum_bins)?;
    let mut planner = RealFftPlanner::<f64>::new();
    let fft = planner.plan_fft_forward(fft_size);
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();

    for (bin, (filter_real, filter_imaginary, length)) in filters.into_iter().enumerate() {
        let pad = fft_size.saturating_sub(filter_real.len()) / 2;
        let end = pad
            .checked_add(filter_real.len())
            .ok_or_else(|| CqtError::InvalidParameters("filter padding overflow".to_owned()))?;
        let scale = length / fft_size as f64;
        let destination = input.get_mut(pad..end).ok_or_else(|| {
            CqtError::InvalidParameters("real filter does not fit FFT input".to_owned())
        })?;
        for (destination, source) in destination.iter_mut().zip(&filter_real) {
            *destination = source * scale;
        }
        fft.process_with_scratch(&mut input, &mut spectrum, &mut scratch)
            .map_err(|source| CqtError::Fft(source.to_string()))?;
        let real_spectrum = spectrum.clone();

        input.fill(0.0);
        let destination = input.get_mut(pad..end).ok_or_else(|| {
            CqtError::InvalidParameters("imaginary filter does not fit FFT input".to_owned())
        })?;
        for (destination, source) in destination.iter_mut().zip(&filter_imaginary) {
            *destination = source * scale;
        }
        fft.process_with_scratch(&mut input, &mut spectrum, &mut scratch)
            .map_err(|source| CqtError::Fft(source.to_string()))?;
        let real_row = real_basis
            .row_mut(bin)
            .ok_or_else(|| CqtError::InvalidParameters("real basis row is missing".to_owned()))?;
        let imaginary_row = imaginary_basis.row_mut(bin).ok_or_else(|| {
            CqtError::InvalidParameters("imaginary basis row is missing".to_owned())
        })?;
        for (((real, imaginary), real_value), imaginary_value) in real_row
            .iter_mut()
            .zip(imaginary_row.iter_mut())
            .zip(real_spectrum.iter())
            .zip(spectrum.iter())
        {
            *real = real_value.re - imaginary_value.im;
            *imaginary = real_value.im + imaginary_value.re;
        }
        sparsify(real_row, imaginary_row);
        input.fill(0.0);
    }
    Ok(FilterBasis {
        real: real_basis,
        imaginary: imaginary_basis,
        fft_size,
    })
}

fn sparsify(real: &mut [f64], imaginary: &mut [f64]) {
    let mut magnitudes = real
        .iter()
        .zip(imaginary.iter())
        .map(|(&real, &imaginary)| real.hypot(imaginary))
        .collect::<Vec<_>>();
    let total = magnitudes.iter().sum::<f64>();
    if total <= f64::MIN_POSITIVE {
        return;
    }
    magnitudes.sort_by(f64::total_cmp);
    let target = total * 0.01;
    let mut cumulative = 0.0;
    let threshold = magnitudes.iter().copied().find(|magnitude| {
        cumulative += magnitude;
        cumulative >= target
    });
    let Some(threshold) = threshold else {
        return;
    };
    for (real, imaginary) in real.iter_mut().zip(imaginary.iter_mut()) {
        if real.hypot(*imaginary) < threshold {
            *real = 0.0;
            *imaginary = 0.0;
        }
    }
}

fn magnitude_response(
    samples: &[f64],
    hop_length: usize,
    basis: &FilterBasis,
    workspace: &mut CqtWorkspace,
) -> Result<Magnitude, CqtError> {
    if basis.real.shape() != basis.imaginary.shape()
        || basis.real.cols() != basis.fft_size / 2 + 1
        || hop_length == 0
    {
        return Err(CqtError::InvalidParameters(
            "filter basis and FFT size disagree".to_owned(),
        ));
    }
    let frame_count = samples
        .len()
        .checked_div(hop_length)
        .and_then(|frames| frames.checked_add(1))
        .ok_or_else(|| CqtError::InvalidParameters("frame count overflow".to_owned()))?;
    let mut response = Magnitude::zeros(basis.real.rows(), frame_count)?;
    let fft = workspace.fft_planner.plan_fft_forward(basis.fft_size);
    let mut scratch = fft.make_scratch_vec();
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let padding = basis.fft_size / 2;

    for frame in 0..frame_count {
        let frame_start = frame
            .checked_mul(hop_length)
            .ok_or_else(|| CqtError::InvalidParameters("frame offset overflow".to_owned()))?;
        for (offset, destination) in input.iter_mut().enumerate() {
            let padded_index = frame_start
                .checked_add(offset)
                .ok_or_else(|| CqtError::InvalidParameters("sample offset overflow".to_owned()))?;
            *destination = padded_index
                .checked_sub(padding)
                .and_then(|index| samples.get(index))
                .copied()
                .unwrap_or(0.0);
        }
        fft.process_with_scratch(&mut input, &mut spectrum, &mut scratch)
            .map_err(|source| CqtError::Fft(source.to_string()))?;
        for bin in 0..basis.real.rows() {
            let real_basis = basis.real.row(bin).ok_or_else(|| {
                CqtError::InvalidParameters("real basis row is missing".to_owned())
            })?;
            let imaginary_basis = basis.imaginary.row(bin).ok_or_else(|| {
                CqtError::InvalidParameters("imaginary basis row is missing".to_owned())
            })?;
            let (real, imaginary) = real_basis.iter().zip(imaginary_basis).zip(&spectrum).fold(
                (0.0, 0.0),
                |(real, imaginary), ((&br, &bi), value)| {
                    (
                        real + br * value.re - bi * value.im,
                        imaginary + br * value.im + bi * value.re,
                    )
                },
            );
            let output = response
                .row_mut(bin)
                .and_then(|row| row.get_mut(frame))
                .ok_or_else(|| {
                    CqtError::InvalidParameters("CQT response index overflow".to_owned())
                })?;
            *output = real.hypot(imaginary);
        }
    }
    Ok(response)
}

fn stack_octaves(responses: &[(Magnitude, usize)]) -> Result<Magnitude, CqtError> {
    let frame_count = responses
        .iter()
        .map(|(magnitude, _)| magnitude.cols())
        .min()
        .ok_or_else(|| CqtError::InvalidParameters("CQT produced no octaves".to_owned()))?;
    let mut output = Magnitude::zeros(FREQUENCY_BINS, frame_count)?;
    let mut row_offset = 0_usize;
    for (magnitude, octave_bins) in responses.iter().rev() {
        let skip = magnitude.rows().saturating_sub(*octave_bins);
        let bins_to_copy = (*octave_bins).min(FREQUENCY_BINS.saturating_sub(row_offset));
        for bin in 0..bins_to_copy {
            let source = magnitude.row(skip.saturating_add(bin)).ok_or_else(|| {
                CqtError::InvalidParameters("CQT octave row is missing".to_owned())
            })?;
            let destination = output
                .row_mut(row_offset.saturating_add(bin))
                .ok_or_else(|| {
                    CqtError::InvalidParameters("CQT output row is missing".to_owned())
                })?;
            let source = source.get(..frame_count).ok_or_else(|| {
                CqtError::InvalidParameters("CQT octave has too few frames".to_owned())
            })?;
            destination.copy_from_slice(source);
        }
        row_offset = row_offset.saturating_add(bins_to_copy);
    }
    Ok(output)
}
