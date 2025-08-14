use std::{
    marker::PhantomData,
    time::{Duration, Instant},
    u128,
};

use log::info;

#[derive(Debug, Clone, Copy)]
pub struct HistogramTimeMetrics {
    label: &'static str,
    count: u128,
    avg: f64,
    ema: f64,
    min: u128,
    max: u128,
}

impl HistogramTimeMetrics {
    const EMA_ALPHA: f64 = 2.0 / (10.0 + 1.0);

    pub fn new(label: &'static str) -> Self {
        Self {
            label,
            count: 0,
            avg: 0.0,
            ema: 0.0,
            min: u128::MAX,
            max: u128::MIN,
        }
    }

    pub fn record(&mut self, val: Duration) {
        let micros = val.as_micros();
        self.count += 1;
        self.avg += (micros as f64 - self.avg) / (self.count as f64);

        if self.ema == 0.0 {
            self.ema = micros as f64;
        } else {
            self.ema = Self::EMA_ALPHA * (micros as f64) + (1.0 - Self::EMA_ALPHA) * self.ema;
        }

        self.min = self.min.min(micros);
        self.max = self.max.max(micros);
    }

    pub fn print(&self) -> String {
        format!(
            "({}) avg={:?} | ema={:?} | max={:?} | min={:?} | cnt={}",
            self.label,
            Duration::from_micros(self.avg as u64),
            Duration::from_micros(self.ema as u64),
            Duration::from_micros(self.max as u64),
            Duration::from_micros(self.min as u64),
            self.count
        )
    }
}

pub trait MetricDef: Sized + Copy + Into<usize> {
    const COUNT: usize;
    const LABELS: &'static [&'static str];
}

pub struct Metrics<T: MetricDef> {
    histograms: Vec<HistogramTimeMetrics>,
    last_print: Instant,
    interval: Duration,
    _phantom: PhantomData<T>,
}

impl<T: MetricDef> Metrics<T> {
    pub fn new(interval: Duration) -> Self {
        let histograms = (0..T::COUNT)
            .map(|i| HistogramTimeMetrics::new(T::LABELS[i]))
            .collect();

        Self {
            histograms,
            interval,
            last_print: Instant::now(),
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn get_mut(&mut self, metric: T) -> &mut HistogramTimeMetrics {
        &mut self.histograms[metric.into()]
    }

    pub fn batch_print(&mut self) {
        if self.last_print.elapsed() < self.interval {
            return;
        }

        let mut output = "[Metrics]".to_string();
        for h in &self.histograms {
            output.push_str("\n\t");
            output.push_str(&h.print());
        }

        self.last_print = Instant::now();
        info!("{output}");
    }
}

#[macro_export]
macro_rules! define_metrics {
    (
        $(#[$outer:meta])*
        $vis:vis enum $enum_name:ident {
            $($variant:ident => $label:literal),* $(,)?
        }
    ) => {
        $(#[$outer])*
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        #[repr(usize)]
        $vis enum $enum_name {
            $($variant),*
        }

        impl From<$enum_name> for usize {
            fn from(val: $enum_name) -> Self {
                val as usize
            }
        }

        // This implementation correctly provides the data needed
        // by `Metrics::new` to build the Vec.
        impl $crate::metrics::MetricDef for $enum_name {
            const COUNT: usize = [$($enum_name::$variant),*].len();
            const LABELS: &'static [&'static str] = &[$($label),*];
        }
    };
}
