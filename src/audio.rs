use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use symphonia::core::audio::SampleBuffer;
use symphonia::core::codecs::CODEC_TYPE_NULL;
use symphonia::core::errors::Error as SymphoniaError;
use symphonia::core::io::MediaSourceStream;
use symphonia::core::probe::Hint;
use tracing::warn;

/// Decode an audio file and mix its channels to mono PCM.
pub fn load_mono(path: &Path) -> Result<(Vec<f32>, u32)> {
    let file = File::open(path).with_context(|| format!("could not open {}", path.display()))?;
    let source = MediaSourceStream::new(Box::new(file), Default::default());
    let mut hint = Hint::new();
    if let Some(extension) = path.extension().and_then(|extension| extension.to_str()) {
        hint.with_extension(extension);
    }
    let probed = symphonia::default::get_probe()
        .format(&hint, source, &Default::default(), &Default::default())
        .context("could not probe audio format")?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|track| track.codec_params.codec != CODEC_TYPE_NULL)
        .context("audio file contains no decodable track")?;
    let track_id = track.id;
    let sample_rate = track
        .codec_params
        .sample_rate
        .context("audio track has no sample rate")?;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &Default::default())
        .context("could not create audio decoder")?;
    let mut mono = Vec::new();

    loop {
        let packet = match format.next_packet() {
            Ok(packet) => packet,
            Err(SymphoniaError::IoError(source))
                if source.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                break;
            }
            Err(source) => return Err(source).context("could not read audio packet"),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(decoded) => decoded,
            Err(SymphoniaError::DecodeError(message)) => {
                warn!(message, "skipping corrupt audio packet");
                continue;
            }
            Err(source) => return Err(source).context("could not decode audio packet"),
        };
        let channels = decoded.spec().channels.count();
        if channels == 0 {
            anyhow::bail!("decoded audio packet has no channels");
        }
        let mut samples = SampleBuffer::<f32>::new(decoded.capacity() as u64, *decoded.spec());
        samples.copy_interleaved_ref(decoded);
        mono.extend(
            samples
                .samples()
                .chunks_exact(channels)
                .map(|frame| frame.iter().copied().sum::<f32>() / channels as f32),
        );
    }
    if mono.is_empty() {
        anyhow::bail!("audio track decoded to no samples");
    }
    Ok((mono, sample_rate))
}
