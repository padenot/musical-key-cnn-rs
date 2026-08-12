use std::fmt;

use serde::Serialize;

use crate::{Error, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum KeyMode {
    Minor,
    Major,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CamelotKey {
    number: u8,
    mode: KeyMode,
}

impl CamelotKey {
    pub fn new(number: u8, mode: KeyMode) -> Result<Self> {
        if !(1..=12).contains(&number) {
            return Err(Error::InvalidModelOutput(format!(
                "Camelot number {number} is outside 1..=12"
            )));
        }
        Ok(Self { number, mode })
    }

    pub(crate) fn from_class(class: usize) -> Result<Self> {
        let class = u8::try_from(class).map_err(|source| {
            Error::InvalidModelOutput(format!("key class does not fit u8: {source}"))
        })?;
        match class {
            0..=11 => Self::new(class + 1, KeyMode::Minor),
            12..=23 => Self::new(class - 11, KeyMode::Major),
            _ => Err(Error::InvalidModelOutput(format!(
                "key class {class} is outside 0..=23"
            ))),
        }
    }

    #[must_use]
    pub const fn number(self) -> u8 {
        self.number
    }

    #[must_use]
    pub const fn mode(self) -> KeyMode {
        self.mode
    }
}

impl fmt::Display for CamelotKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let suffix = match self.mode {
            KeyMode::Minor => 'A',
            KeyMode::Major => 'B',
        };
        write!(formatter, "{}{suffix}", self.number)
    }
}

impl Serialize for CamelotKey {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.collect_str(self)
    }
}

#[cfg(test)]
mod tests {
    use super::CamelotKey;

    #[test]
    fn maps_upstream_classes_to_camelot() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(CamelotKey::from_class(0)?.to_string(), "1A");
        assert_eq!(CamelotKey::from_class(11)?.to_string(), "12A");
        assert_eq!(CamelotKey::from_class(12)?.to_string(), "1B");
        assert_eq!(CamelotKey::from_class(23)?.to_string(), "12B");
        Ok(())
    }
}
