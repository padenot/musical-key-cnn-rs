#![cfg(all(target_os = "macos", feature = "coreml"))]

use std::path::Path;

use anyhow::{Context, Result, ensure};
use beat_this::RustnnCoremlModel;
use beat_this::{Model, RtenRuntime, Runtime, Tensor};

const OUTPUTS: [(&str, &str); 22] = [
    (
        "/conv1/conv/Conv_output_0",
        "_conv1_conv_Conv_output_0_conv2d",
    ),
    ("/conv1/elu/Elu_output_0", "_conv1_elu_Elu_output_0"),
    (
        "/conv2/conv/Conv_output_0",
        "_conv2_conv_Conv_output_0_conv2d",
    ),
    ("/conv2/elu/Elu_output_0", "_conv2_elu_Elu_output_0"),
    ("/pool1/MaxPool_output_0", "_pool1_MaxPool_output_0"),
    (
        "/conv3/conv/Conv_output_0",
        "_conv3_conv_Conv_output_0_conv2d",
    ),
    ("/conv3/elu/Elu_output_0", "_conv3_elu_Elu_output_0"),
    (
        "/conv4/conv/Conv_output_0",
        "_conv4_conv_Conv_output_0_conv2d",
    ),
    ("/conv4/elu/Elu_output_0", "_conv4_elu_Elu_output_0"),
    ("/pool2/MaxPool_output_0", "_pool2_MaxPool_output_0"),
    (
        "/conv5/conv/Conv_output_0",
        "_conv5_conv_Conv_output_0_conv2d",
    ),
    ("/conv5/elu/Elu_output_0", "_conv5_elu_Elu_output_0"),
    (
        "/conv6/conv/Conv_output_0",
        "_conv6_conv_Conv_output_0_conv2d",
    ),
    ("/conv6/elu/Elu_output_0", "_conv6_elu_Elu_output_0"),
    ("/pool3/MaxPool_output_0", "_pool3_MaxPool_output_0"),
    (
        "/conv7/conv/Conv_output_0",
        "_conv7_conv_Conv_output_0_conv2d",
    ),
    ("/conv7/elu/Elu_output_0", "_conv7_elu_Elu_output_0"),
    (
        "/conv8/conv/Conv_output_0",
        "_conv8_conv_Conv_output_0_conv2d",
    ),
    ("/conv8/elu/Elu_output_0", "_conv8_elu_Elu_output_0"),
    (
        "/conv9/conv/Conv_output_0",
        "_conv9_conv_Conv_output_0_conv2d",
    ),
    ("/conv9/elu/Elu_output_0", "_conv9_elu_Elu_output_0"),
    (
        "/global_avgpool/GlobalAveragePool_output_0",
        "_global_avgpool_GlobalAveragePool_output_0",
    ),
];

fn input() -> Tensor {
    let shape = vec![1, 1, 105, 512];
    let data = (0..(105 * 512))
        .map(|index| ((index % 257) as f32 - 128.0) / 256.0)
        .collect();
    Tensor { shape, data }
}

#[test]
#[ignore = "diagnostic model exposes large intermediate tensors"]
fn coreml_intermediates_match_rten() -> Result<()> {
    let input = input();
    let mut rten = RtenRuntime.load_model(Path::new("models/keynet-debug.onnx"))?;
    let coreml_names = OUTPUTS.map(|(_, coreml)| coreml);
    let mut coreml = RustnnCoremlModel::load_with_additional_outputs(
        Path::new("models/keynet.json"),
        &coreml_names,
    )?;
    let expected_outputs = rten.run(&[("spectrogram", &input)])?;
    let actual_outputs = coreml.run(&[("spectrogram", &input)])?;
    let mut overall_maximum = 0.0_f32;

    for (rten_name, coreml_name) in OUTPUTS {
        let expected = expected_outputs
            .get(rten_name)
            .with_context(|| format!("RTen returned no {rten_name}"))?;
        let actual = actual_outputs
            .get(coreml_name)
            .with_context(|| format!("Core ML returned no {coreml_name}"))?;
        ensure!(
            expected.shape == actual.shape,
            "{coreml_name} shapes differ: RTen {:?}, Core ML {:?}",
            expected.shape,
            actual.shape
        );
        let (maximum, sum) = expected.data.iter().zip(&actual.data).fold(
            (0.0_f32, 0.0_f64),
            |(maximum, sum), (expected, actual)| {
                let difference = (expected - actual).abs();
                (maximum.max(difference), sum + f64::from(difference))
            },
        );
        overall_maximum = overall_maximum.max(maximum);
        let mean = sum / expected.data.len() as f64;
        eprintln!("{coreml_name}: max={maximum:.8} mean={mean:.8}");
    }

    ensure!(
        overall_maximum < 1e-3,
        "Core ML intermediate maximum difference is {overall_maximum}"
    );
    Ok(())
}
