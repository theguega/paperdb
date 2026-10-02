//! Calendar days. Only built from the clock or by arithmetic, so always a valid date.

use std::time::{SystemTime, UNIX_EPOCH};

/// Days since 1970-01-01 (UTC).
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct Day(i64);

impl Day {
    #[must_use]
    pub fn today() -> Self {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |d| d.as_secs());
        Self(i64::try_from(secs / 86_400).unwrap_or(0))
    }

    #[must_use]
    pub fn minus(self, days: u32) -> Self {
        Self(self.0 - i64::from(days))
    }

    /// `YYYY-MM-DD`
    #[must_use]
    pub fn iso(self) -> String {
        let (y, m, d) = self.ymd();
        format!("{y:04}-{m:02}-{d:02}")
    }

    /// `YYYYMMDD`, the form the arXiv query API wants.
    #[must_use]
    pub fn compact(self) -> String {
        let (y, m, d) = self.ymd();
        format!("{y:04}{m:02}{d:02}")
    }

    /// Howard Hinnant's `civil_from_days`.
    fn ymd(self) -> (i64, i64, i64) {
        let z = self.0 + 719_468;
        let era = z.div_euclid(146_097);
        let doe = z.rem_euclid(146_097);
        let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = doy - (153 * mp + 2) / 5 + 1;
        let m = if mp < 10 { mp + 3 } else { mp - 9 };
        (yoe + era * 400 + i64::from(m <= 2), m, d)
    }
}

#[cfg(test)]
mod tests {
    use super::Day;

    #[test]
    fn civil() {
        assert_eq!(Day(0).iso(), "1970-01-01");
        assert_eq!(Day(20_728).iso(), "2026-10-02");
        assert_eq!(Day(11_016).compact(), "20000229");
    }
}
