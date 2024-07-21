{%- include "structs.wgsl" -%}

@group(0) @binding(0)
var<storage, read> input_0: Array; // data

@group(0) @binding(1)
var<storage, read_write> output_0: Array;

@compute @workgroup_size({{ workgroup_size_x }})
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
	let gidx = global_id.x;
	let x0 = {{ i_shape[0][0] }}i;
	let x1 = {{ i_shape[0][1] }}i;

	{%- if i_shape[0] | length == 3 -%}
		let x2 = {{ i_shape[0][2] }}i;
		let x3 = 1i;
	{%- elif i_shape[0] | length == 4 -%}
		let x2 = {{ i_shape[0][2] }}i;
		let x3 = {{ i_shape[0][3] }}i;
	{%- else -%}
		let x2 = 1i;
		let x3 = 1i;
	{%- endif -%}

	let a = i32(gidx) / (x1 * x2 * x3);
	let b = (i32(gidx) % (x1 * x2 * x3)) / (x2 * x3);
	let c = (i32(gidx) % (x2 * x3)) / x3;
	let d = i32(gidx) % x3;

	// Assume that starts, ends, axes and steps are only 1 element	
	let start = {{ starts[0] }}i;
	var end = {{ ends[0] }}i;
	let ax = {{ axes[0] }}i;
	let step = {{ steps[0] }}i;

	// I'm not sure this if statement is moved to the compiler or not
	{% if i_shape[0] | length == 4 %}
		let end2 = {{ i_shape[0][2] }}i;
		let end3 = {{ i_shape[0][3] }}i;
	{% elif i_shape[0] | length == 3 %}
		let end2 = {{ i_shape[0][2] }}i;
		let end3 = 0i;
	{% else %}
		let end2 = 0i; // unreachable
		let end3 = 0i; // unreachable
	{% endif %}
	
	if end == 2147483647i {
		if ax == 0 {
			end = {{ i_shape[0][0] }}i;
		} else if ax == 1 {
			end = {{ i_shape[0][1] }}i;
		} else if ax == 2 {
			end = end2;
		} else if ax == 3 {
			end = end3;
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
			let ceil = (end - start + step - 1) / step;
			if reminder == 0 {
				let j = (b - start) / step;
				let idx = a * ceil * x2 * x3 
					+ j * x2 * x3 
					+ c * x3 
					+ d;
				output_0.data[idx] = input_0.data[gidx];
			}
		}
	} else if ax == 2 {
		if start <= c && c < end {
			let reminder = (c - start) % step;
			let ceil = (end - start + step - 1) / step;
			if reminder == 0 {
				let j = (c - start) / step;
				let idx = a * x1 * ceil * x3 
					+ b * ceil * x3 
					+ j * x3 
					+ d;
				output_0.data[idx] = input_0.data[gidx];
			}
		}
	} else if ax == 3 {
		if start <= d && d < end {
			let reminder = (d - start) % step;
			let ceil = (end - start + step - 1) / step;
			if reminder == 0 {
				let j = (d - start) / step;
				let idx = a * x1 * x2 * ceil
					+ b * x2 * ceil
					+ c * ceil
					+ j;
				output_0.data[idx] = input_0.data[gidx];
			}
		}
	}
}
