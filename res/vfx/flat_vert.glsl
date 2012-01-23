////////////////////////////////////////////////////////////////////////////////
//
// Flat Shader
// ===========
//
// Vertex Shader
//
// File:              /res/vfx/flat_vert.glsl
// Author:            Martin Kirst
// Creation Date:     2012.01.06
// Description:
//
// Simple shader without lighting.
//
////////////////////////////////////////////////////////////////////////////////


#version 330


////////////////////////////////////////////////////////////////////////////////
// Variables
////////////////////////////////////////////////////////////////////////////////


//layout(std140) uniform StandardUniforms {
	//mat4 projection;
	//mat4 view;
	//mat4 model;
//};

uniform mat4 ProjectionView;
in vec4 in_Position;
in vec3 in_Normal;

out vec3 out_vs_worldPos;
out vec3 out_vs_normal;

////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void main() {
	//gl_Position = projection * view * model * in_Position;

	out_vs_worldPos=in_Position.xyz;
	out_vs_normal = in_Normal;

	gl_Position =ProjectionView* in_Position;
}
