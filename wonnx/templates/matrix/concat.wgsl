
{%- include "structs.wgsl" -%}

{% for input in i_lens %}

@group({{ loop.index0 / 4 | int }}) @binding({{ loop.index0 % 4}})
var<storage, read> input_{{ loop.index0 }}: Array;

{% endfor %}

{% set binding_len = i_lens | length %}
@group({{ binding_len  / 4 | int }}) @binding({{ binding_len % 4 }})
var<storage, read_write> output_0: Array;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(num_workgroups) num_workgroups: vec3<u32>) {
	let gidx = global_id.x;
	let gidy = global_id.y;

	let x_executions = num_workgroups.x * 16u;

	let actual_idx = gidx + gidy * x_executions;
	
	let ax = {{ axis }}u;

	{% for input in i_lens %}
		{% if loop.first %}
			if (actual_idx < {{ i_lens[0] }}u) {
				var global_input_idx = actual_idx;
				var input_indices: array<u32, {{ i_shape[0] | length }}> = array<u32, {{ i_shape[0] | length }}>();
				{% set in_shape = i_shape[0] %}
				var input_shape: array<u32, {{ i_shape[0] | length }}> = array<u32, {{ i_shape[0] | length }}>({{ in_shape | join(sep=', ')}});
				let output_shape_length = {{ o_shape[0] | length }}u;

				// calculate input indices
				for (var i = 0u; i < output_shape_length; i = i + 1u) {
					input_indices[output_shape_length - i - 1] = global_input_idx % input_shape[output_shape_length - i - 1];
					global_input_idx = global_input_idx / input_shape[output_shape_length - i - 1];
				}

				// calculate output index 
				{% set out_shape = o_shape[0] %}
				var output_shape = array<u32, {{ o_shape[0] | length }}>({{ out_shape | join(sep=', ')}});

				var output_idx = 0u;
				for (var i = 0u; i < output_shape_length; i = i + 1u) {
					output_idx = output_idx * output_shape[i] + input_indices[i];
				}
				output_0.data[output_idx] = input_0.data[actual_idx];
			}

		{% else %}
			if ((actual_idx >= {{ cum_len | nth(n=loop.index0 -1) }}u) && (actual_idx < {{ cum_len | nth(n=loop.index0)}}u)) {				
				var global_input_idx = actual_idx  - {{ cum_len | nth(n=loop.index0 -1) }}u;
				var input_indices: array<u32, {{ i_shape[0] | length }}> = array<u32, {{ i_shape[0] | length }}>();
				{% set shape = i_shape[loop.index0] %}
				var input_shape: array<u32, {{ i_shape[0] | length }}> = array<u32, {{ i_shape[0] | length }}>({{ shape | join(sep=', ')}});
				let output_shape_length = {{ o_shape[0] | length }}u;
				
				// calculate input indices
				var input_idx = global_input_idx;
				for (var i = 0u; i < output_shape_length; i = i + 1u) {
					input_indices[output_shape_length - i - 1] = input_idx % input_shape[output_shape_length - i - 1];
					input_idx = input_idx / input_shape[output_shape_length - i - 1];
				}


				// calculate the offset
				var offset = 0u;
				{% for i in range(end=loop.index0) %}
					offset = offset + {{ i_shape[i][axis] }}u;
				{% endfor %}

				// add offset to input indices
				input_indices[ax] = input_indices[ax] + offset;

				// calculate output index
				{% set out_shape = o_shape[0] %}
				var output_shape = array<u32, {{ o_shape[0] | length }}>({{ out_shape | join(sep=', ')}});

				var output_idx = 0u;
				for (var i = 0u; i < output_shape_length; i = i + 1u) {
					output_idx = output_idx * output_shape[i] + input_indices[i];
				}
				
				output_0.data[output_idx] = input_{{ loop.index0 }}.data[global_input_idx];
			}

		{% endif %}
	{% endfor %}
}
