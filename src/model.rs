#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GbModel {
    Dmg0,
    Dmg,
    Mgb,
    Sgb,
    Sgb2,
    Cgb0,
    Cgb,
    Agb,
}

impl GbModel {
    pub fn is_cgb(self) -> bool {
        matches!(self, GbModel::Cgb0 | GbModel::Cgb | GbModel::Agb)
    }

    pub fn is_agb(self) -> bool {
        self == GbModel::Agb
    }

    pub fn is_sgb(self) -> bool {
        self == GbModel::Sgb || self == GbModel::Sgb2
    }

    /// CPU clock rate in Hz. SGB1 runs at SNES master clock / 5 (~2.4% faster).
    pub fn cpu_clock_rate(self) -> u32 {
        match self {
            GbModel::Sgb => 4_295_454, // 21_477_272 / 5
            _ => 4_194_304,            // 2^22
        }
    }
}

impl std::fmt::Display for GbModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GbModel::Dmg0 => write!(f, "dmg0"),
            GbModel::Dmg => write!(f, "dmg"),
            GbModel::Mgb => write!(f, "mgb"),
            GbModel::Sgb => write!(f, "sgb"),
            GbModel::Sgb2 => write!(f, "sgb2"),
            GbModel::Cgb0 => write!(f, "cgb0"),
            GbModel::Cgb => write!(f, "cgb"),
            GbModel::Agb => write!(f, "agb"),
        }
    }
}

impl std::str::FromStr for GbModel {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "dmg0" => Ok(GbModel::Dmg0),
            "dmg" => Ok(GbModel::Dmg),
            "mgb" => Ok(GbModel::Mgb),
            "sgb" => Ok(GbModel::Sgb),
            "sgb2" => Ok(GbModel::Sgb2),
            "cgb0" => Ok(GbModel::Cgb0),
            "cgb" | "gbc" => Ok(GbModel::Cgb),
            "agb" | "gba" => Ok(GbModel::Agb),
            _ => Err(format!(
                "unknown model '{}' (expected: dmg0, dmg, mgb, sgb, sgb2, cgb0, cgb/gbc, agb/gba)",
                s
            )),
        }
    }
}
