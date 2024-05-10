use image::imageops;
use image::{imageops::FilterType, ImageBuffer, Pixel, Rgb};
use imageproc::drawing::draw_hollow_rect_mut;
use imageproc::rect::Rect;
use log::info;
use std::collections::HashMap;
use std::convert::TryInto;
use std::time::Instant;
use std::vec;
use std::{
    fs,
    io::{BufRead, BufReader},
    path::Path,
};
use wonnx::utils::OutputTensor;
use wonnx::WonnxError;

fn draw_rect(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, x1: f32, y1: f32, x2: f32, y2: f32) {
    let x1 = x1 as u32;
    let y1 = y1 as u32;
    let x2 = x2 as u32;
    let y2 = y2 as u32;
    let rect = Rect::at(x1 as i32, y1 as i32).of_size((x2 - x1) as u32, (y2 - y1) as u32);
    draw_hollow_rect_mut(image, rect, Rgb([255, 0, 0]));
}

fn padding_image(image: ImageBuffer<Rgb<u8>, Vec<u8>>) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
    let (width, height) = image.dimensions();
    let target_size = if width > height { width } else { height };
    let mut new_image = ImageBuffer::new(target_size as u32, target_size as u32);
    let x_offset = (target_size as u32 - width) / 2;
    let y_offset = (target_size as u32 - height) / 2;
    for j in 0..height {
        for i in 0..width {
            let pixel = image.get_pixel(i, j);
            new_image.put_pixel(i + x_offset, j + y_offset, *pixel);
        }
    }
    new_image
}

fn load_image() -> Vec<f32> {
    let args: Vec<String> = std::env::args().collect();
    let image_path = if args.len() == 2 {
        Path::new(&args[1]).to_path_buf()
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../data/images")
            .join("dog.jpg")
    };

    let image_buffer: ImageBuffer<Rgb<u8>, Vec<u8>> = image::open(image_path).unwrap().to_rgb8();
    let image_buffer = padding_image(image_buffer);
    let mut image_buffer = imageops::resize(&image_buffer, 416, 416, FilterType::Nearest);
    draw_rect(&mut image_buffer, 0., 0., 100., 200.);
    image_buffer.save("resized_dog.jpg").unwrap();

    // convert image to Vec<f32> with channel first format
    let mut image = vec![0.0; 3 * 416 * 416];
    for j in 0..416 {
        for i in 0..416 {
            let pixel = image_buffer.get_pixel(i as u32, j as u32);
            let channels = pixel.channels();
            for c in 0..3 {
                image[c * 416 * 416 + j * 416 + i] = channels[c] as f32;
            }
        }
    }
    return image;
}

fn get_coco_labels() -> Vec<String> {
    // Download the ImageNet class labels, matching SqueezeNet's classes.
    let labels_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../data/models")
        .join("coco-classes.txt");
    let file = BufReader::new(fs::File::open(labels_path).unwrap());

    file.lines().map(|line| line.unwrap()).collect()
}

// Args Management
async fn run() {
    // Output shape is [1, 3549, 85]
    // 85 = 4 (bounding box) + 1 (objectness) + 80 (class probabilities)
    let outputs = execute_gpu().await.unwrap();
    let output = outputs.get("output").unwrap();
    let output: &[f32] = output.try_into().unwrap();
    let labels = get_coco_labels();

    let mut detections = vec![];
    for i in 0..3549 {
        let offset = i * 85;
        let objectness = output[offset + 4];

        let (class, score) = output[offset + 5..offset + 85]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let class = labels[class].clone();
        let x1 = output[offset];
        let y1 = output[offset + 1];
        let x2 = output[offset + 2];
        let y2 = output[offset + 3];
        detections.push((class, objectness, score, x1, y1, x2, y2));
    }

    // Filter out low objectness detections
    detections.retain(|(_, objectness, _, _, _, _, _)| *objectness > 0.2);
    detections.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (class, objectness, score, x1, y1, x2, y2) in detections.iter() {
        println!(
            "class: {}, objectness: {}, score: {}, x1: {}, y1: {}, x2: {}, y2: {}",
            class, objectness, score, x1, y1, x2, y2
        );
    }
}

// Hardware management
async fn execute_gpu() -> Result<HashMap<String, OutputTensor>, WonnxError> {
    let mut input_data = HashMap::new();
    // let image = load_input();
    let image = load_image();
    let images = image.as_slice().try_into().unwrap();
    input_data.insert("images".to_string(), images);

    let model_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../data/models")
        .join("yolox_nano.onnx");
    let session = wonnx::Session::from_path(model_path).await?;
    let time_pre_compute = Instant::now();
    info!("Start Compute");
    let result = session.run(&input_data).await?;
    let time_post_compute = Instant::now();
    println!(
        "time: first_prediction: {:#?}",
        time_post_compute - time_pre_compute
    );
    Ok(result)
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
        let time_pre_compute = Instant::now();

        pollster::block_on(run());
        let time_post_compute = Instant::now();
        println!("time: main: {:#?}", time_post_compute - time_pre_compute);
    }
    #[cfg(target_arch = "wasm32")]
    {
        // std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        //  console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
    }
}
