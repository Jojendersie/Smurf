////////////////////////////////////////////////////////////////////////////////
//
// Flat Shader
// ===========
//
// Fragment Shader
//
// File:              /res/vfx/flat_frag.glsl
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


//uniform vec4 solidColor;
uniform vec3 eyePos;
uniform mat4 ProjectionView;

out vec4 out_Color;

in vec3 out_vs_normal;
in vec3 out_vs_worldPos;


////////////////////////////////////////////////////////////////////////////////
// Functions
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void main() {
	vec3 lightPos=vec3(3,5,-1);
	float kAmbient=0.6f,kDiffuse=0.4f,kSpecular=0.8f,specPower=10.0,LightPower=5.8;;

	vec3 rayView=normalize(out_vs_worldPos-eyePos);
	vec3 rayLight=normalize(out_vs_worldPos-lightPos);
	float diffuse=clamp(dot(rayLight,out_vs_normal),0,1);
	float shine=clamp(dot(max(reflect(rayLight,out_vs_normal),0),rayView),0,1);

	float phongShade=max(diffuse*kDiffuse+pow(shine,specPower)*kSpecular*LightPower,0)+kAmbient;

	vec3 color= vec3(1,0.5,0)*phongShade;
	out_Color = vec4(color,1);
}
