use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

pub(super) fn resample(
    samples: &[f64],
    input_rate: u32,
    output_rate: u32,
) -> Result<Vec<f64>, String> {
    if samples.is_empty() {
        return Err("input is empty".to_owned());
    }
    if input_rate == 0 || output_rate == 0 {
        return Err("sample rates must be positive".to_owned());
    }
    if input_rate == output_rate {
        return Ok(samples.to_vec());
    }

    let ratio = f64::from(output_rate) / f64::from(input_rate);
    let expected_length = (samples.len() as f64 * ratio).ceil() as usize;
    let mut resampler = Fft::<f64>::new(
        input_rate as usize,
        output_rate as usize,
        1_024,
        1,
        FixedSync::Input,
    )
    .map_err(|source| source.to_string())?;
    let input =
        InterleavedSlice::new(samples, 1, samples.len()).map_err(|source| source.to_string())?;
    let mut output = resampler
        .process_all(&input, samples.len(), None)
        .map_err(|source| source.to_string())?
        .take_data();
    output.truncate(expected_length);
    output.resize(expected_length, 0.0);
    Ok(output)
}
