{%- include "structs.wgsl" -%}

struct Indices {
	data: array<i32>
};

@group(0) @binding(0)
var<storage, read> input_0: Array; // data

@group(0) @binding(1)
var<storage, read> input_1: Indices; // starts

@group(0) @binding(2)
var<storage, read> input_2: Indices; // ends

{% if defined_axes and defined_steps %}
	@group(0) @binding(3)
	var<storage, read> input_3: Indices; // axes

	@group(1) @binding(0)
	var<storage, read> input_4: Indices; // steps

	@group(1) @binding(1)
	var<storage, read_write> output_0: Array;
{% elif defined_axes and not defined_steps %}
	@group(0) @binding(3)
	var<storage, read> input_3: Indices; // axes

	@group(1) @binding(0)
	var<storage, read_write> output_0: Array;
{% elif not defined_axes and defined_steps %}
	@group(0) @binding(3)
	var<storage, read> input_3: Indices; // steps

	@group(1) @binding(0)
	var<storage, read_write> output_0: Array;
{% else %}
	@group(0) @binding(3)
	var<storage, read_write> output_0: Array;
{% endif %}


@compute @workgroup_size({{ workgroup_size_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;
	let x0 = {{ i_shape[0][0] }}i;
	let x1 = {{ i_shape[0][1] }}i;

	{%- if i_shape[0] | length >= 3 -%}
		let x2 = {{ i_shape[0][2] }}i;
	{%- else -%}
		let x2 = 1i;
	{%- endif -%}

	{%- if i_shape[0] | length >= 4 -%}
		let x3 = {{ i_shape[0][3] }}i;
	{%- else -%}
		let x3 = 1i;
	{%- endif -%}

	{%- if i_shape[0] | length == 1 -%}
		let ax0_end = {{ i_shape[0][0] }}i;
		let ax1_end = 0i;
		let ax2_end = 0i;
		let ax3_end = 0i;
	{%- elif i_shape[0] | length == 2 -%}
		let ax0_end = {{ i_shape[0][0] }}i;
		let ax1_end = {{ i_shape[0][1] }}i;
		let ax2_end = 0i;
		let ax3_end = 0i;
	{%- elif i_shape[0] | length == 3 -%}
		let ax0_end = {{ i_shape[0][0] }}i;
		let ax1_end = {{ i_shape[0][1] }}i;
		let ax2_end = {{ i_shape[0][2] }}i;
		let ax3_end = 0i;
	{%- elif i_shape[0] | length == 4 -%}
		let ax0_end = {{ i_shape[0][0] }}i;
		let ax1_end = {{ i_shape[0][1] }}i;
		let ax2_end = {{ i_shape[0][2] }}i;
		let ax3_end = {{ i_shape[0][3] }}i;
	{%- endif -%}

	let a = i32(gidx) / (x1 * x2 * x3);
	let b = (i32(gidx) % (x1 * x2 * x3)) / (x2 * x3);
	let c = (i32(gidx) % (x2 * x3)) / x3;
	let d = i32(gidx) % x3;

	// Assume that starts, ends, axes and steps are only 1 element	

	let start = input_1.data[0];
	var end = input_2.data[0];

	{%- if defined_axes and defined_steps -%}
		let ax = input_3.data[0];
		let step = input_4.data[0];
	{%- elif defined_axes and not defined_steps -%}
		let ax = input_3.data[0];
		let step = 1;
	{%- elif not defined_axes and defined_steps -%}
		let ax = 0;
		let step = input_3.data[0];
	{%- else -%}
		let ax = 0;
		let step = 1;
	{%- endif -%}

	if end == 2147483647i {
		if ax == 0 {
			end = ax0_end;
		} else if ax == 1 {
			end = ax1_end;
		} else if ax == 2 {
			end = ax2_end;
		} else if ax == 3 {
			end = ax3_end;
		}
	}

	if ax == 0 {
		if start <= a && a < end {
			let reminder = (a - start) % step;
			if reminder == 0 {
				let j = (a - start) / step;
				let idx = j * x1 * x2 * x3 
					+ b * x2 * x3 
					+ c * x3 
					+ d;
				output_0.data[idx] = input_0.data[gidx];
			}
		}
	} else if ax == 1 {
		if start <= b && b < end {
			let reminder = (b - start) % step;
			if reminder == 0 {
				let j = (b - start) / step;
				let idx = a * (end - start) / step * x2 * x3 
					+ j * x2 * x3 
					+ c * x3 
					+ d;
				output_0.data[idx] = input_0.data[gidx];
			}
		}
	} else if ax == 2 {
		if start <= c && c < end {
			let reminder = (c - start) % step;
			if reminder == 0 {
				let j = (c - start) / step;
				let idx = a * x1 * (end - start) / step * x3 
					+ b * (end - start) / step * x3 
					+ j * x3 
					+ d;
				output_0.data[idx] = input_0.data[gidx];
			}
		}
	} else if ax == 3 {
		if start <= d && d < end {
			let reminder = (d - start) % step;
			if reminder == 0 {
				let j = (d - start) / step;
				let idx = a * x1 * x2 * (end - start) / step 
					+ b * x2 * (end - start) / step 
					+ c * (end - start) / step 
					+ j;
				output_0.data[idx] = input_0.data[gidx];
			}
		}
	}
}
