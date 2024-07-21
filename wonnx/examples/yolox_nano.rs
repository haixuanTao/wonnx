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
use wonnx::WonnxError;

/*-----------------------------------------------------------------------------
 Post processing
--------------------------------------------------------------------------------*/
fn draw_rect(image: &mut ImageBuffer<Rgb<u8>, Vec<u8>>, x1: f32, y1: f32, x2: f32, y2: f32) {
    let x1 = x1 as u32;
    let y1 = y1 as u32;
    let x2 = x2 as u32;
    let y2 = y2 as u32;
    let rect = Rect::at(x1 as i32, y1 as i32).of_size(x2 - x1 as u32, (y2 - y1) as u32);
    draw_hollow_rect_mut(image, rect, Rgb([255, 0, 0]));
}

fn calc_loc(positions: &Vec<(f32, f32, f32, f32)>) -> Vec<(f32, f32, f32, f32)> {
    let mut locs = vec![];

    // calc girds
    let (h, w) = (416, 416);
    let strides = vec![8, 16, 32];
    let mut h_grids = vec![];
    let mut w_grids = vec![];

    for stride in strides.iter() {
        let mut h_grid = vec![0.0; h / stride];
        let mut w_grid = vec![0.0; w / stride];

        for i in 0..h / stride {
            h_grid[i] = i as f32;
        }
        for i in 0..w / stride {
            w_grid[i] = i as f32;
        }
        h_grids.push(h_grid);
        w_grids.push(w_grid);
    }
    let acc = vec![0, 52 * 52, 52 * 52 + 26 * 26, 52 * 52 + 26 * 26 + 13 * 13];

    for (i, stride) in strides.iter().enumerate() {
        let h_grid = &h_grids[i];
        let w_grid = &w_grids[i];
        let idx = acc[i];

        for (i, y) in h_grid.iter().enumerate() {
            for (j, x) in w_grid.iter().enumerate() {
                let p = idx + i * w / stride + j;
                let (px, py, pw, ph) = positions[p];
                let (x, y) = ((x + px) * *stride as f32, (y + py) * *stride as f32);
                let (ww, hh) = (pw.exp() * *stride as f32, ph.exp() * *stride as f32);
                let loc = (x - ww / 2.0, y - hh / 2.0, x + ww / 2.0, y + hh / 2.0);
                locs.push(loc);
            }
        }
    }
    locs
}

fn non_max_suppression(
    boxes: &Vec<(f32, f32, f32, f32)>,
    scores: &Vec<f32>,
    score_threshold: f32,
    iou_threshold: f32,
) -> Vec<(usize, (f32, f32, f32, f32))> {
    let mut new_boxes = vec![];
    let mut sorted_indices = (0..boxes.len()).collect::<Vec<_>>();
    sorted_indices.sort_by(|a, b| scores[*a].partial_cmp(&scores[*b]).unwrap());

    while let Some(last) = sorted_indices.pop() {
        let mut remove_list = vec![];
        let score = scores[last];
        let bbox = boxes[last];
        let mut numerator = (
            bbox.0 * score,
            bbox.1 * score,
            bbox.2 * score,
            bbox.3 * score,
        );
        let mut denominator = score;

        for i in 0..sorted_indices.len() {
            let idx = sorted_indices[i];
            let (x1, y1, x2, y2) = boxes[idx];
            let (x1_, y1_, x2_, y2_) = boxes[last];
            let box1_area = (x2 - x1) * (y2 - y1);

            let inter_x1 = x1.max(x1_);
            let inter_y1 = y1.max(y1_);
            let inter_x2 = x2.min(x2_);
            let inter_y2 = y2.min(y2_);
            let inter_w = (inter_x2 - inter_x1).max(0.0);
            let inter_h = (inter_y2 - inter_y1).max(0.0);
            let inter_area = inter_w * inter_h;
            let area1 = (x2 - x1) * (y2 - y1);
            let area2 = (x2_ - x1_) * (y2_ - y1_);
            let union_area = area1 + area2 - inter_area;
            let iou = inter_area / union_area;

            if scores[idx] < score_threshold {
                remove_list.push(i);
            } else if iou > iou_threshold {
                remove_list.push(i);
                let w = scores[idx] * iou;
                numerator = (
                    numerator.0 + boxes[idx].0 * w,
                    numerator.1 + boxes[idx].1 * w,
                    numerator.2 + boxes[idx].2 * w,
                    numerator.3 + boxes[idx].3 * w,
                );
                denominator += w;
            } else if inter_area / box1_area > 0.7 {
                remove_list.push(i);
            }
        }
        for i in remove_list.iter().rev() {
            sorted_indices.remove(*i);
        }
        let new_bbox = (
            numerator.0 / denominator,
            numerator.1 / denominator,
            numerator.2 / denominator,
            numerator.3 / denominator,
        );
        new_boxes.push((last, new_bbox));
    }
    new_boxes
}

fn post_process(preds: &[f32]) -> Vec<(String, f32, f32, f32, f32, f32)> {
    let labels = get_coco_labels();
    let mut positions = vec![];
    let mut classes = vec![];
    let mut objectnesses = vec![];
    for i in 0..3549 {
        let offset = i * 85;
        let objectness = preds[offset + 4];

        let (class, score) = preds[offset + 5..offset + 85]
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        let class = labels[class].clone();
        let x1 = preds[offset];
        let y1 = preds[offset + 1];
        let x2 = preds[offset + 2];
        let y2 = preds[offset + 3];
        classes.push((class, score));
        positions.push((x1, y1, x2, y2));
        objectnesses.push(objectness);
    }

    let locs = calc_loc(&positions);

    let mut result = vec![];
    // filter by objectness
    let indices = non_max_suppression(&locs, &objectnesses, 0.5, 0.3);
    for bbox in indices {
        let (i, (x, y, w, h)) = bbox;
        let (class, &score) = &classes[i];
        result.push((class.clone(), score, x, y, w, h));
    }
    result
}

/*-----------------------------------------------------------------------------
 Pre processing
--------------------------------------------------------------------------------*/
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

fn load_image() -> (Vec<f32>, ImageBuffer<Rgb<u8>, Vec<u8>>) {
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
    let image_buffer = imageops::resize(&image_buffer, 416, 416, FilterType::Nearest);

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
    return (image, image_buffer);
}

fn get_coco_labels() -> Vec<String> {
    // Download the ImageNet class labels, matching SqueezeNet's classes.
    let labels_path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../data/models")
        .join("coco-classes.txt");
    let file = BufReader::new(fs::File::open(labels_path).unwrap());

    file.lines().map(|line| line.unwrap()).collect()
}

/*-----------------------------------------------------------------------------
 Main
--------------------------------------------------------------------------------*/
// Hardware management
async fn execute_gpu() -> Result<Vec<(String, f32, f32, f32, f32, f32)>, WonnxError> {
    let mut input_data = HashMap::new();
    let (image, _) = load_image();
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

    info!("Start Post Processing");
    let time_pre_compute = Instant::now();
    let output = result.get("output").unwrap();
    let output = output.try_into().unwrap();
    let positions = post_process(output);
    let time_post_compute = Instant::now();
    println!(
        "time: post_processing: {:#?}",
        time_post_compute - time_pre_compute
    );

    Ok(positions)
}

async fn run() {
    // Output shape is [1, 3549, 85]
    // 85 = 4 (bounding box) + 1 (objectness) + 80 (class probabilities)
    let preds = execute_gpu().await.unwrap();

    let (_, image_buffer) = load_image();
    let mut image_buffer = image_buffer;
    for (class, score, x0, y0, x1, y1) in preds.iter() {
        println!(
            "class: {}, score: {}, x0: {}, y0: {}, x1: {}, y1: {}",
            class, *score, *x0, *y0, *x1, *y1
        );
        draw_rect(&mut image_buffer, *x0, *y0, *x1, *y1);
    }
    image_buffer.save("yolox_predict.jpg").unwrap();
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
