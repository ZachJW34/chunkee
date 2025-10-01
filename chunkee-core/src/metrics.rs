use std::{
    fmt::{self, Display, Write},
    marker::PhantomData,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
    time::{Duration, Instant},
    u64,
};

use log::info;
use once_cell::sync::Lazy;

static APP_START: Lazy<Instant> = Lazy::new(Instant::now);

pub trait Metric: Default + Display {
    fn new(label: &'static str) -> Self;
}

#[derive(Default)]
pub struct HistogramMetrics {
    label: &'static str,
    count: AtomicU64,
    total_time: AtomicU64,
    min: AtomicU64,
    max: AtomicU64,
    ema: AtomicU64,
}

impl Metric for HistogramMetrics {
    fn new(label: &'static str) -> Self {
        Self {
            label,
            min: AtomicU64::new(u64::MAX),
            ..Default::default()
        }
    }
}

impl HistogramMetrics {
    const EMA_ALPHA: f64 = 2.0 / (10.0 + 1.0);

    pub fn record(&self, val: Duration) {
        let micros = val.as_micros() as u64;
        self.count.fetch_add(1, Relaxed);
        self.total_time.fetch_add(micros, Relaxed);
        self.min.fetch_min(micros, Relaxed);
        self.max.fetch_max(micros, Relaxed);
        let _ = self.ema.fetch_update(Relaxed, Relaxed, |ema| {
            if ema == 0 {
                Some(micros)
            } else {
                let ema_f =
                    Self::EMA_ALPHA * (micros as f64) + (1.0 - Self::EMA_ALPHA) * ema as f64;
                Some(ema_f as u64)
            }
        });
    }
}

impl Display for HistogramMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.count.load(Relaxed);
        let total_time = self.total_time.load(Relaxed);
        let min = self.min.load(Relaxed);
        let max = self.max.load(Relaxed);
        let ema = self.ema.load(Relaxed);
        let avg = total_time.checked_div(count).unwrap_or(0);
        let rate =
            (count as f64) / (total_time as f64) * (Duration::from_secs(1).as_micros() as f64);

        write!(
            f,
            "({}) avg={:?} | ema={:?} | max={:?} | min={:?} | total={:?} | cnt={} | rate={}/s",
            self.label,
            Duration::from_micros(avg),
            Duration::from_micros(ema),
            Duration::from_micros(max),
            Duration::from_micros(min),
            Duration::from_micros(total_time),
            count,
            rate as u64,
        )
    }
}

#[derive(Default)]
pub struct ThroughputMetrics {
    label: &'static str,
    start: AtomicU64,
    active: AtomicU64,
    count: AtomicU64,
}

impl Metric for ThroughputMetrics {
    fn new(label: &'static str) -> Self {
        Self {
            label,
            ..Default::default()
        }
    }
}

impl ThroughputMetrics {
    pub fn start(&self) {
        let now = APP_START.elapsed().as_millis() as u64;
        let _ = self.start.compare_exchange(0, now, Relaxed, Relaxed);
    }

    pub fn record(&self) {
        self.count.fetch_add(1, Relaxed);
    }

    pub fn end(&self) {
        let start = self.start.swap(0, Relaxed);
        if start != 0 {
            let elapsed_duration = APP_START.elapsed() - Duration::from_millis(start);
            self.active
                .fetch_add(elapsed_duration.as_millis() as u64, Relaxed);
        }
    }
}

impl Display for ThroughputMetrics {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let count = self.count.load(Relaxed);
        let active_ms = self.active.load(Relaxed);
        let active_duration = Duration::from_millis(active_ms);

        let throughput = if active_ms > 0 {
            (count as f64 / active_duration.as_secs_f64()) as u64
        } else {
            0
        };

        write!(
            f,
            "({}) throughput={}/s | active_time={:?} | count={}",
            self.label, throughput, active_duration, count,
        )
    }
}

pub trait MetricDef: Sized + Copy + Into<usize> {
    const COUNT: usize;
    const LABELS: &'static [&'static str];
}

pub struct MetricsRegistry<E: MetricDef, M: Metric> {
    metrics: Vec<M>,
    _phantom: PhantomData<E>,
}

impl<E: MetricDef, M: Metric> MetricsRegistry<E, M> {
    pub fn new() -> Self {
        let metrics = (0..E::COUNT).map(|i| M::new(E::LABELS[i])).collect();

        Self {
            metrics,
            _phantom: PhantomData,
        }
    }

    #[inline(always)]
    pub fn get(&self, metric_def: E) -> &M {
        &self.metrics[metric_def.into()]
    }
}

impl<E: MetricDef, M: Metric> Display for MetricsRegistry<E, M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for h in &self.metrics {
            write!(f, "\n\t{}", h)?;
        }
        Ok(())
    }
}

pub struct MetricsPrinter {
    last_print: Instant,
    interval: Duration,
}

impl MetricsPrinter {
    pub fn new(interval: Duration) -> Self {
        Self {
            interval,
            last_print: Instant::now(),
        }
    }

    pub fn batch_print(&mut self, registries: &[&dyn Display]) {
        if self.last_print.elapsed() < self.interval {
            return;
        }

        let mut output = "[Metrics]".to_string();
        for registry in registries {
            let _ = write!(output, "{}", registry);
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

        impl $crate::metrics::MetricDef for $enum_name {
            const COUNT: usize = [$($enum_name::$variant),*].len();
            const LABELS: &'static [&'static str] = &[$($label),*];
        }
    };
}
