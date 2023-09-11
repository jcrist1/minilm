use std::{error::Error, sync::Arc};

use anyhow::Context;
use leptos::*;
use log::{debug, info};
use minilm::{Cpu, CpuError, MiniLM};
use serde::{Deserialize, Serialize};

fn main() {
    _ = console_log::init_with_level(log::Level::Debug);
    console_error_panic_hook::set_once();
    mount_to_body(move |cx| {
        view! { cx,  <App />}
    })
}

async fn get_data(path: &str) -> anyhow::Result<Vec<u8>> {
    let data = reqwasm::http::Request::get(path)
        .send()
        .await
        .context("Failed to get ${path}")?
        .binary()
        .await
        .context("Failed to get data from ${path}")?;

    let str_data = String::from_utf8_lossy(&data[..100]);
    debug!("Got data: {str_data}");
    Ok(data)
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct SiteData {
    tokenizer_data: Vec<u8>,
    model_data: Vec<u8>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
enum SiteRes<T> {
    Ok(T),
    Err(SiteError),
}

impl<T> SiteRes<T> {
    fn res(self) -> Result<T, SiteError> {
        match self {
            SiteRes::Ok(t) => Ok(t),
            SiteRes::Err(err) => Err(err),
        }
    }
}

impl<T> From<anyhow::Result<T>> for SiteRes<T> {
    fn from(value: anyhow::Result<T>) -> Self {
        match value {
            Ok(t) => SiteRes::Ok(t),
            Err(e) => SiteRes::Err(e.into()),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
struct SiteError(String);

impl std::fmt::Display for SiteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

impl From<anyhow::Error> for SiteError {
    fn from(value: anyhow::Error) -> Self {
        SiteError(format!(
            "{value}, {}",
            value
                .chain()
                .map(|err| format!("{err}"))
                .collect::<String>()
        ))
    }
}

impl Error for SiteError {}

async fn get_minilm_data() -> anyhow::Result<SiteData> {
    let tokenizer_data = get_data("/tokenizer.json")
        .await
        .context("Failed to get tokenizer data")?;
    let model_data = get_data("/model.safetensors")
        .await
        .context("Failed to get model data")?;
    Ok(SiteData {
        tokenizer_data,
        model_data,
    })
}

trait Dot<Target> {
    fn dot(&self, t: &Target) -> f32;
}

impl<H: AsRef<[f32]>, T: AsRef<[f32]>> Dot<T> for H {
    fn dot(&self, t: &T) -> f32 {
        self.as_ref()
            .iter()
            .zip(t.as_ref().iter())
            .map(|(x, y)| *x * *y)
            .sum::<f32>()
    }
}

fn score_vecs<'a, H: AsRef<[f32]>, T: AsRef<[f32]>>(
    left: Result<&'a H, &'a CpuError>,
    right: Result<&'a T, &'a CpuError>,
) -> Result<f32, &'a CpuError> {
    let left = left?;
    let right = right?;
    Ok(left.dot(right))
}

#[component]
fn Loaded(cx: Scope, #[prop()] model: Arc<MiniLM<f32, Cpu>>) -> impl IntoView {
    let (left_vec_data, set_left_vec_data) = create_signal(cx, Ok(vec![0.0; 384]));
    let (right_vec_data, set_right_vec_data) = create_signal(cx, Ok(vec![0.0; 384]));
    let (score, set_score) = create_signal(cx, Ok(0.0));
    let model_for_right = model.clone();
    let on_input_right = move |ev| {
        let model = Arc::clone(&model_for_right);
        let text = event_target_value(&ev);
        let right = if &text == "Boop" {
            Err(CpuError::WrongNumElements)
        } else {
            model.encode(&text)
        };

        let left = left_vec_data();
        let new_score = (score_vecs(left.as_ref(), right.as_ref())).map_err(|err| *err);
        set_right_vec_data(right);
        set_score(new_score);
    };

    let model_for_left = model.clone();
    let on_input_left = move |ev| {
        let model = Arc::clone(&model_for_left);
        let text = event_target_value(&ev);
        let left = if &text == "Boop" {
            Err(CpuError::WrongNumElements)
        } else {
            model.encode(&text)
        };

        let right = right_vec_data();
        let new_score = (score_vecs(left.as_ref(), right.as_ref())).map_err(|err| *err);
        set_left_vec_data(left);
        set_score(new_score);
    };

    view! {cx,
        <p>"Type something to encode"</p>
        <p>
        <input on:input=on_input_left /> <input on:input=on_input_right />
        </p>
        <p>
        <ErrorBoundary
            fallback = |_cx, _errors| view! {cx,  "Failed to encode text"}
        >
            <p>"Successfully encoded text"</p>
            <p>"Similarity Score: " {score}</p>
        </ErrorBoundary>
        </p>
    }
}

fn load_model(site_res: SiteRes<SiteData>) -> anyhow::Result<MiniLM<f32, Cpu>> {
    let data = site_res.res()?;
    Ok(MiniLM::new(&data.tokenizer_data, &data.model_data)?)
}

#[component]
fn App(cx: Scope) -> impl IntoView {
    let async_data = create_resource(
        cx,
        || (),
        |_| async move {
            let data: SiteRes<SiteData> = get_minilm_data().await.into();
            data
        },
    );

    view! { cx,
            <h1> MiniLM Similarity Scorer</h1>
            {move || {
                let data = async_data.read(cx).map(load_model);
                match data {
                 Some(Ok(model)) => {
                     view! { cx,
                        <Loaded model=Arc::new(model)/>
                     }
                 }
                 Some(Err(err)) => {
                     error!("Error loading model: {err}");
                     View::Text(view!{cx, "Could not encode text"})
                 }
                 None => {
                     info!("Data not loaded yet");
                     View::Text(view!{cx, "Loading"})

                 }
             }}

        }
    }
}
