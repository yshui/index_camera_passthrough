#version 450

layout(binding = 0) uniform Parameters {
    // Distortion coefficients
    vec4 dcoef;
    // Optical center
    vec2 center;
    // Focal length in terms of focal divided by sensor_width
    vec2 focal;
    // Scaling of the output image
    vec2 scale;
    // Pixel size of the sensor_width
    float sensorSize;
    vec2 texOffset;
};
layout(binding = 1) uniform sampler2D inputTex;
in vec4 gl_FragCoord;

// Input coordinates -0.5 ~ 0.5
// relative to the center of the undistorted image,
// this way we align the optical center to the center of the image.
layout(location = 0) in noperspective vec2 coord;
layout(location = 0) out vec4 outColor;
void main() {
    vec2 r = coord * scale / focal;
    // Also scale the r so the whole circular region will be included
    // in the output.
    float theta = atan(length(r));
    float theta2 = theta * theta;
    theta *= 1 + theta2 * (dcoef.x +
                 theta2 * (dcoef.y +
                 theta2 * (dcoef.z +
                 theta2 * dcoef.w)));
    // Scale r vector to length theta
    vec2 mapped = theta / length(r) * r;
    mapped *= focal;
    // mapped should now be -0.5~0.5, in inputTex coord
    // move mapped so its centered at `center`
    mapped = mapped + center;
    // mapped is now 0 ~ 1
    // scale x by 0.5 because inputTex is 2 image side by side
    mapped.x *= 0.5;
    // mapped is now (0~0.5, 0~1.0);
    outColor = texture(inputTex, mapped + texOffset);
}
