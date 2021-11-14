#version 450
layout(binding = 1) uniform sampler2D tex;
layout(binding = 2) uniform Info {
	vec2 texOffset;
};
layout(location = 0) in vec4 gl_FragCoord;
layout(location = 1) in noperspective vec3 texCoord;
layout(location = 0) out vec4 color;

void main() {
	// Perspective divide here, if we do this in vertex
	// shader the texCoord won't be interpolated correctly
	// because of perspective.
	vec2 tex_coord = texCoord.xy / texCoord.z;
	tex_coord = tex_coord + vec2(0.25, 0.5);

	if (tex_coord.x < 0 || tex_coord.x > 0.5) {
		color = vec4(0.0, 0.0, 0.0, 0.0);
	} else {
		tex_coord = tex_coord + texOffset;
		tex_coord.y = 1.0 - tex_coord.y;
		color = texture(tex, tex_coord);
	}
}
