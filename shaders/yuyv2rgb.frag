#version 450
in vec4 gl_FragCoord;
layout(binding = 0) uniform sampler2D yuyvtex;
layout(location = 0) out vec4 color;
// bt709 -> rgb conversion matrix
const mat3 yuv_matrix = mat3(
	1.164,  1.164, 1.164,
	0.000, -0.392, 2.017,
	1.596, -0.813, 0.000
);

void main() {
	vec2 coord = gl_FragCoord.xy - vec2(0.5, 0.5);
	vec2 size = vec2(textureSize(yuyvtex, 0));
	vec2 tex_coord = vec2(floor(coord.x / 2.0) + 0.5, coord.y + 0.5) / size;
	vec4 yuyv = texture(yuyvtex, tex_coord);
	vec3 yuv;
	if (mod(coord.x, 2.0) == 0) {
		yuv = vec3(yuyv.xyw);
	} else {
		yuv = vec3(yuyv.zyw);
	}
	yuv -= vec3(0.0625, 0.5, 0.5);
	color = vec4(yuv_matrix * yuv, 1.0);
}
