use std::{
    ffi::OsStr,
    fs::File,
    io::Read,
    path::{Path, PathBuf},
};

use derive_more::{Add, From, Into, Sub};
use serde::{Deserialize, Serialize};
use variant_count::VariantCount;

use crate::{
    params::{
        GeneralEnvParams, ModulationBank, ModulationSend, ModulationType, OSCParams, Parameters,
        LFO,
    },
    sound_gen::{Decibel, FilterParams, NoteShapeDiscrim},
};

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PresetData {
    pub name: String,
    pub master_vol: Decibel,
    pub osc_1: PresetDataOSC,
    pub osc_2: PresetDataOSC,
    pub osc_2_mod: ModulationType,
    pub mod_bank: PresetDataModBank,
}

impl PresetData {
    pub fn from_params(params: &Parameters, name: String) -> Self {
        PresetData {
            name,
            master_vol: params.master_vol,
            osc_1: PresetDataOSC::from(&params.osc_1),
            osc_2: PresetDataOSC::from(&params.osc_2),
            osc_2_mod: params.osc_2_mod,
            mod_bank: PresetDataModBank::from(&params.mod_bank),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PresetDataOSC {
    pub volume: Decibel,
    pub pan: f32,
    pub phase: f32,
    pub coarse_tune: I32Divable,
    pub fine_tune: f32,
    pub shape: NoteShapeDiscrim,
    pub warp: f32,
    pub vol_attack: f32,
    pub vol_hold: f32,
    pub vol_decay: f32,
    pub vol_sustain: Decibel,
    pub vol_release: f32,
    pub pitch_attack: f32,
    pub pitch_hold: f32,
    pub pitch_decay: f32,
    pub pitch_multiply: f32,
    pub pitch_release: f32,
    pub vol_lfo: LFO,
    pub pitch_lfo: LFO,
    pub filter: PresetDataFilter,
}

impl From<&OSCParams> for PresetDataOSC {
    fn from(params: &OSCParams) -> Self {
        PresetDataOSC {
            volume: params.volume,
            phase: params.phase,
            pan: params.pan,
            shape: params.shape.get_shape(),
            warp: params.shape.get_warp(),
            coarse_tune: I32Divable::from(params.coarse_tune),
            fine_tune: params.fine_tune,
            vol_attack: params.vol_adsr.attack,
            vol_hold: params.vol_adsr.hold,
            vol_decay: params.vol_adsr.decay,
            vol_sustain: params.vol_adsr.sustain,
            vol_release: params.vol_adsr.release,
            pitch_attack: params.pitch_adsr.attack,
            pitch_hold: params.pitch_adsr.hold,
            pitch_decay: params.pitch_adsr.decay,
            pitch_multiply: params.pitch_adsr.multiply,
            pitch_release: params.pitch_adsr.release,
            vol_lfo: params.vol_lfo,
            pitch_lfo: params.pitch_lfo,
            filter: PresetDataFilter::from(&params.filter_params),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PresetDataModBank {
    pub env_1_send: ModulationSend,
    pub env_1: GeneralEnvParams,
    pub env_2_send: ModulationSend,
    pub env_2: GeneralEnvParams,
}

impl From<&ModulationBank> for PresetDataModBank {
    fn from(params: &ModulationBank) -> Self {
        PresetDataModBank {
            env_1: params.env_1,
            env_1_send: params.env_1_send.mod_type,
            env_2: params.env_2,
            env_2_send: params.env_2_send.mod_type,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PresetDataFilter {
    pub filter_type: FilterTypeDiscrim,
    pub freq: f32,
    pub q: f32,
    pub db_gain: f32,
}

impl From<&FilterParams> for PresetDataFilter {
    fn from(params: &FilterParams) -> Self {
        PresetDataFilter {
            filter_type: FilterTypeDiscrim::from(&params.filter),
            freq: params.freq,
            q: params.q_value,
            db_gain: get_db_gain(params.filter),
        }
    }
}

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

/// Try to parse a given file into a `PresetData`
pub fn try_parse_file(file: impl AsRef<Path>) -> Result<PresetData, Box<dyn std::error::Error>> {
    let mut file = std::fs::File::open(file)?;
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    Ok(serde_json::from_str(&contents)?)
}

/// Read a folder and try to parse any `json` files in it into a PresetData.
pub fn get_presets_from_folder(folder: impl AsRef<Path>) -> std::io::Result<Vec<PathBuf>> {
    let mut presets = vec![];
    for entry in (std::fs::read_dir(folder)?).flatten() {
        if Some(std::ffi::OsStr::new("json")) == entry.path().extension() {
            presets.push(entry.path());
        }
    }
    Ok(presets)
}

/// Save a PresetData to the given path. This returns an error if the file could
/// not be saved or if serialization fails for any reason.
pub fn save_preset_to_file(
    preset: PresetData,
    path: impl AsRef<Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let file = File::create(&path)?;
    Ok(serde_json::to_writer_pretty(file, &preset)?)
}

/// Get a file name which is not currently used by any file in the given folder
/// The file will have the file name and extention provided.
pub fn get_free_file_name(
    folder: impl AsRef<Path>,
    file_name: impl AsRef<OsStr>,
    extention: impl AsRef<OsStr>,
) -> (PathBuf, usize) {
    let mut i = 0;
    loop {
        let mut path = folder.as_ref().to_path_buf();
        path.push(format!("{} {}", file_name.as_ref().to_str().unwrap(), i));
        path.set_extension(&extention);
        if !path.exists() {
            return (path, i);
        }
        i += 1;
    }
}
