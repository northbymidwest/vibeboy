#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GbModel {
    Dmg0,
    Dmg,
    Mgb,
    Sgb,
    Sgb2,
    Cgb,
}

impl GbModel {
    pub fn is_cgb(self) -> bool {
        self == GbModel::Cgb
    }

    pub fn is_sgb(self) -> bool {
        self == GbModel::Sgb || self == GbModel::Sgb2
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
            GbModel::Cgb => write!(f, "cgb"),
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
            "cgb" | "gbc" => Ok(GbModel::Cgb),
            _ => Err(format!(
                "unknown model '{}' (expected: dmg0, dmg, mgb, sgb, sgb2, cgb/gbc)",
                s
            )),
        }
    }
}
