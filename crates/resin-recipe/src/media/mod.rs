//! CPU media preprocessing shared by native recipes. Pixels are RGB/HWC in [0, 1].

use base64::Engine;
use image::{ImageBuffer, ImageFormat, Rgb, imageops::FilterType};

use crate::Error;

pub mod deepseek;
pub mod gemma;
pub mod glm;
pub mod inkling;
pub mod qwen;

#[derive(Clone, Debug)]
pub struct Tensor<T> {
    pub shape: Vec<usize>,
    pub data: Vec<T>,
}

impl<T> Tensor<T> {
    pub fn new(shape: Vec<usize>, data: Vec<T>) -> Result<Self, Error> {
        if shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)) != Some(data.len()) {
            return Err(Error::new("media tensor shape does not match its storage"));
        }
        Ok(Self { shape, data })
    }
}

#[derive(Clone, Debug)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub data: Vec<f32>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize, data: Vec<f32>) -> Result<Self, Error> {
        if width == 0
            || height == 0
            || width.checked_mul(height).and_then(|n| n.checked_mul(3)) != Some(data.len())
        {
            return Err(Error::new(
                "images must have nonempty RGB/HWC shape [height, width, 3]",
            ));
        }
        if data
            .iter()
            .any(|v| !v.is_finite() || !(0.0..=1.0).contains(v))
        {
            return Err(Error::new("images must contain finite values in [0, 1]"));
        }
        Ok(Self {
            width,
            height,
            data,
        })
    }

    pub fn decode(data: &[u8], media_type: &str) -> Result<Self, Error> {
        if data.is_empty() {
            return Err(Error::new("image bytes must not be empty"));
        }
        let format = match media_type.to_ascii_lowercase().as_str() {
            "image/png" => ImageFormat::Png,
            "image/jpeg" => ImageFormat::Jpeg,
            "image/webp" => ImageFormat::WebP,
            "image/gif" => ImageFormat::Gif,
            "image/bmp" => ImageFormat::Bmp,
            "image/tiff" => ImageFormat::Tiff,
            _ => {
                return Err(Error::new(format!(
                    "unsupported image media type: {media_type:?}"
                )));
            }
        };
        let decoded = image::load_from_memory_with_format(data, format)
            .map_err(|error| Error::new(format!("invalid image: {error}")))?
            .to_rgb32f();
        Self::new(
            decoded.width() as usize,
            decoded.height() as usize,
            decoded.into_raw(),
        )
    }

    pub fn data_url(source: &str) -> Result<Self, Error> {
        let (header, encoded) = source.split_once(",").ok_or_else(|| {
            Error::new("image URLs must be base64 data URLs; remote fetching is disabled")
        })?;
        let media_type = header
            .strip_prefix("data:")
            .and_then(|s| s.strip_suffix(";base64"))
            .ok_or_else(|| {
                Error::new("image URLs must be base64 data URLs; remote fetching is disabled")
            })?;
        let data = base64::engine::general_purpose::STANDARD
            .decode(encoded)
            .map_err(|_| Error::new("image data URL contains invalid base64"))?;
        Self::decode(&data, media_type)
    }

    pub fn resize(&self, width: usize, height: usize, filter: FilterType) -> Result<Self, Error> {
        let dims = [self.width, self.height, width, height].map(u32::try_from);
        let [Ok(sw), Ok(sh), Ok(dw), Ok(dh)] = dims else {
            return Err(Error::new("image dimensions exceed u32"));
        };
        if dw == 0 || dh == 0 {
            return Err(Error::new("resized image dimensions must be positive"));
        }
        if (sw, sh) == (dw, dh) {
            return Ok(self.clone());
        }
        let source = ImageBuffer::<Rgb<f32>, _>::from_raw(sw, sh, self.data.as_slice())
            .ok_or_else(|| Error::new("invalid RGB image storage"))?;
        let resized = image::imageops::resize(&source, dw, dh, filter);
        Ok(Self {
            width,
            height,
            data: resized.into_raw(),
        })
    }
}

/// Torchvision's bicubic resize, which is what HF's image processors use.
///
/// The kernel is **not** the difference: torchvision's cubic coefficient is -0.5, the same
/// Catmull-Rom the `image` crate's `FilterType::CatmullRom` uses, and a float reimplementation
/// of the separable form matches `tvF.resize` on a float tensor to 0.0000. What differs is
/// that HF's pipeline resizes **uint8** tensors, so torchvision quantizes to 8 bits **between
/// the two separable passes** as well as at the end. Without that intermediate rounding the
/// cubic's overshoot survives to the second pass, and on a 2.5x upscale of a black circle on
/// white the two disagree by up to 18 levels of 255 - 0.14 in the normalized tensor, which is
/// not rounding.
///
/// With the intermediate quantization this reproduces torchvision on the measured branches:
/// 0 differing pixels on a 1000x64 -> 1024x96 resize and on 300x500 -> 288x512, and 21 of
/// 196608 on 100x100 -> 256x256 (tie rounding). The source and the result are [0, 1] f32,
/// which is what `RgbImage` holds; the arithmetic runs in 8-bit levels.
pub fn resize_torchvision_bicubic(
    image: &RgbImage,
    width: usize,
    height: usize,
) -> Result<RgbImage, Error> {
    const A: f64 = -0.5; // Catmull-Rom, and torchvision's bicubic
    fn cubic(x: f64) -> f64 {
        let x = x.abs();
        if x < 1.0 {
            ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
        } else if x < 2.0 {
            ((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A
        } else {
            0.0
        }
    }
    /// (source index, weight) per output sample: the support widens by the reduction factor,
    /// which is what makes a downscale antialias, and the weights are normalized.
    fn weights(input: usize, output: usize) -> Vec<Vec<(usize, f64)>> {
        let scale = input as f64 / output as f64;
        let filter_scale = scale.max(1.0);
        (0..output)
            .map(|i| {
                let center = scale * (i as f64 + 0.5);
                let low = (center - 2.0 * filter_scale + 0.5).floor().max(0.0) as usize;
                let high = (center + 2.0 * filter_scale - 0.5)
                    .ceil()
                    .min(input as f64) as usize;
                let mut row: Vec<(usize, f64)> = (low..high)
                    .map(|j| (j, cubic((j as f64 + 0.5 - center) / filter_scale)))
                    .collect();
                let sum: f64 = row.iter().map(|(_, v)| v).sum();
                if sum != 0.0 {
                    for (_, v) in &mut row {
                        *v /= sum;
                    }
                }
                row
            })
            .collect()
    }
    let dims = [image.width, image.height, width, height].map(u32::try_from);
    let [Ok(sw), Ok(sh), Ok(dw), Ok(dh)] = dims else {
        return Err(Error::new("image dimensions exceed u32"));
    };
    if dw == 0 || dh == 0 {
        return Err(Error::new("resized image dimensions must be positive"));
    }
    if (sw, sh) == (dw, dh) {
        return Ok(image.clone());
    }
    // levels, as torchvision's uint8 path sees them
    // round, not truncate: `v * 255.0` for a level stored as f32 can land just below the
    // integer, and truncating there turns 100 into 99 before the filter has run
    let source: Vec<f64> = image
        .data
        .iter()
        .map(|v| (v * 255.0).round() as f64)
        .collect();
    // half-to-even, which is what torchvision's uint8 cast does; `round()` rounds halves away
    // from zero and leaves a one-level difference on the ties
    let quantize = |v: f64| v.round_ties_even().clamp(0.0, 255.0);
    let wx = weights(image.width, width);
    let wy = weights(image.height, height);
    let mut horizontal = vec![0.0f64; image.height * width * 3];
    for y in 0..image.height {
        for (x, row) in wx.iter().enumerate() {
            for &(src, weight) in row {
                for c in 0..3 {
                    horizontal[(y * width + x) * 3 + c] +=
                        source[(y * image.width + src) * 3 + c] * weight;
                }
            }
        }
    }
    // the intermediate is 8-bit in torchvision, and that is the whole difference
    for value in &mut horizontal {
        *value = quantize(*value);
    }
    let mut data = vec![0.0f32; height * width * 3];
    for (y, row) in wy.iter().enumerate() {
        for x in 0..width {
            for c in 0..3 {
                let mut acc = 0.0f64;
                for &(src, weight) in row {
                    acc += horizontal[(src * width + x) * 3 + c] * weight;
                }
                data[(y * width + x) * 3 + c] = (quantize(acc) / 255.0) as f32;
            }
        }
    }
    Ok(RgbImage {
        width,
        height,
        data,
    })
}

#[derive(Clone, Debug)]
pub struct Video {
    pub frames: Vec<RgbImage>,
    pub fps: f64,
    pub frame_indices: Vec<usize>,
}

#[derive(Clone, Debug)]
pub enum Source {
    Image(RgbImage),
    Video(Video),
}

pub fn resize_filter(value: u32) -> Result<FilterType, Error> {
    match value {
        0 => Ok(FilterType::Nearest),
        1 => Ok(FilterType::Lanczos3),
        2 => Ok(FilterType::Triangle),
        3 => Ok(FilterType::CatmullRom),
        _ => Err(Error::new("unsupported image resampling filter")),
    }
}
