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
}
