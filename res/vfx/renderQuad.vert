
#version 330


////////////////////////////////////////////////////////////////////////////////
// Variables
////////////////////////////////////////////////////////////////////////////////

in vec3 in_Pos;
in vec2 in_TexCoords;

out vec2 out_vs_texCoords;

////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void main() {
	//gl_Position = projection * view * model * in_Position;
	out_vs_texCoords=in_TexCoords;
	gl_Position = vec4(in_Pos,1);
}
