use derive_more::{Add, From, Into, Sub};
use serde::{Deserialize, Serialize};
use variant_count::VariantCount;

fn get_db_gain(filter: biquad::Type<f32>) -> f32 {
    match filter {
        biquad::Type::LowShelf(db_gain)
        | biquad::Type::HighShelf(db_gain)
        | biquad::Type::PeakingEQ(db_gain) => db_gain,
        _ => 0.0,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, VariantCount, Serialize, Deserialize)]
pub enum FilterTypeDiscrim {
    SinglePoleLowPass,
    LowPass,
    HighPass,
    BandPass,
    Notch,
    AllPass,
    LowShelf,
    HighShelf,
    PeakingEQ,
}

impl<T> From<&biquad::Type<T>> for FilterTypeDiscrim {
    fn from(filter_type: &biquad::Type<T>) -> Self {
        match filter_type {
            biquad::Type::SinglePoleLowPass => FilterTypeDiscrim::SinglePoleLowPass,
            biquad::Type::LowPass => FilterTypeDiscrim::LowPass,
            biquad::Type::HighPass => FilterTypeDiscrim::HighPass,
            biquad::Type::BandPass => FilterTypeDiscrim::BandPass,
            biquad::Type::Notch => FilterTypeDiscrim::Notch,
            biquad::Type::AllPass => FilterTypeDiscrim::AllPass,
            biquad::Type::LowShelf(_) => FilterTypeDiscrim::LowShelf,
            biquad::Type::HighShelf(_) => FilterTypeDiscrim::HighShelf,
            biquad::Type::PeakingEQ(_) => FilterTypeDiscrim::PeakingEQ,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Add, Sub, Serialize, Deserialize, From, Into)]
pub struct I32Divable(i32);

impl std::ops::Div<Self> for I32Divable {
    type Output = f32;

    fn div(self, rhs: Self) -> Self::Output {
        (self.0 as f32) / (rhs.0 as f32)
    }
}

impl std::ops::Mul<f32> for I32Divable {
    type Output = Self;

    fn mul(self, rhs: f32) -> Self::Output {
        I32Divable(((self.0 as f32) * rhs) as i32)
    }
}

impl I32Divable {
    pub const fn new(x: i32) -> I32Divable {
        I32Divable(x)
    }
}
