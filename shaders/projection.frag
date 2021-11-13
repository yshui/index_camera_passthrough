#version 450
layout(binding = 1) uniform sampler2D tex;
layout(binding = 0) uniform Info {
	mat4 mvp;
	vec2 texOffset;
	vec2 windowSize;
	float overlayWidth;
	float eyeOffset;
};
layout(location = 0) in vec4 gl_FragCoord;
layout(location = 0) out vec4 color;

void main() {
	// Normalize Frag Coord
	vec2 coord = gl_FragCoord.xy / windowSize;
	if (coord.x > 1) {
		coord.x = coord.x - 1 - eyeOffset;
	} else {
		coord.x = coord.x + eyeOffset;
	}
	if (coord.x > 1 || coord.x < 0) {
		color = vec4(0.0, 0.0, 0.0, 0.0);
	} else {
		// Center coord
		coord = coord - vec2(0.5, 0.5);
		// Change coordinate system: mvp is y up, gl_FragCoord is y down
		coord.y = -coord.y;
		// Resize coord based on size of the overlay
		coord *= overlayWidth;
		// Project the coord
		vec4 tmp = mvp * vec4(coord, 0, 1);
		vec2 tex_coord = tmp.xy / tmp.z;

		tex_coord = tex_coord + vec2(0.25, 0.5);
		if (tex_coord.x < 0 || tex_coord.x > 0.5) {
			color = vec4(0.0, 0.0, 0.0, 0.0);
		} else {
			tex_coord = tex_coord + texOffset;
			tex_coord.y = 1.0 - tex_coord.y;
			color = texture(tex, tex_coord);
		}
	}
}
