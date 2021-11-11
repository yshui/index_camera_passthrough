#version 450

layout(binding = 0) uniform Parameters {
	// Distortion coefficients
	vec3 dcoef;
	// Optical center for left/right side
	vec2 cleft, cright;
	// Focal length in terms of focal divided by sensor_width
	float focal;
	// Scaling of the output image
	float scale;
};
layout(binding = 1) uniform sampler2D inputTex;
in vec4 gl_FragCoord;
layout(location = 0) out vec4 outColor;
void main() {
	vec2 r;
	vec2 tsize = vec2(textureSize(inputTex, 0));
	float focal_px = tsize.x / 2.0 * focal;
	bool is_left = gl_FragCoord.x < tsize.x / 2.0;
	// Calculate r as vector to the center of the undistorted image,
	// this way we align the optical center to the center of the image.
	// Also scale the r so the whole circular region will be included
	// in the output.
	if (is_left) {
		r = gl_FragCoord.xy - vec2(tsize.x / 4, tsize.y / 2);
	} else {
		r = gl_FragCoord.xy - vec2(tsize.x * 3 / 4, tsize.y / 2);
	}
	r = r * scale;
	r = r / focal_px;
	float r2 = r.x * r.x + r.y * r.y;
	float r4 = r2 * r2;
	float r6 = r2 * r4;
	float factor = 1 + dcoef[0] * r2 + dcoef[1] * r4 + dcoef[2] * r6;
	vec2 mapped = r * factor * focal_px;
	if (is_left) {
		mapped = mapped + cleft;
	} else {
		mapped = mapped + cright + vec2(tsize.x / 2, 0);
	}
	if ((is_left && mapped.x > tsize.x / 2) || (!is_left && mapped.x < tsize.x / 2)) {
		// Don't let image to leak from one eye to another
		outColor = vec4(0.0, 0.0, 0.0, 1.0);
	} else {
		vec2 x = mapped / tsize;
		outColor = texture(inputTex, mapped / tsize);
	}
}
