#version 330


////////////////////////////////////////////////////////////////////////////////
// Variables
////////////////////////////////////////////////////////////////////////////////
uniform sampler2D texSampler;

out vec4 out_Color;

in vec2 out_vs_texCoords;


////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void main() {

	out_Color = vec4(texture(texSampler,out_vs_texCoords).rgb,1);
}
