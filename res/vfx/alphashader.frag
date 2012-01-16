#version 330

uniform mat4 invProjectionView;
uniform vec3 eyePos;
uniform float k;//constant: height_of_the_prism*number_of_particles*constant
uniform float maxTime;
uniform vec3 color;

smooth in vec3 gs_out_normal;
smooth in float gs_out_alphaTime;
smooth in float	gs_out_alphaCurvature;
flat in float gs_out_alphaShape;
flat in float gs_out_area;

out vec4 fragColor;

void main()
{
	vec3 worldPos=(invProjectionView*((gl_FragCoord-vec4(1.0))*2.0)).xyz;
	vec3 viewRay=worldPos-eyePos;
	float gamma=dot(gs_out_normal,viewRay)/(sqrt(dot(gs_out_normal,gs_out_normal))*sqrt(dot(viewRay,viewRay)));

	float alphaDensity=clamp(k/(gs_out_area*gamma),0.0,1.0);
	float alphaFade=clamp(1.0-gs_out_alphaTime/maxTime,0.0,1.0);

	fragColor= vec4(color,alphaDensity*gs_out_alphaShape*gs_out_alphaCurvature*alphaFade);
}