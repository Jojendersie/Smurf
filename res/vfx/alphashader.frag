#version 330

uniform mat4 invProjectionView;
uniform vec3 eyePos;
uniform float k;//constant: height_of_the_prism*number_of_particles*constant
uniform float maxTime;
uniform vex3 color;

smooth in vec3 out_normal;
smooth in float out_alphaTime;
smooth in float	out_alphaCurvature;
flat in float alphaShape;
flat in float area;

out vec4 fragColor;

void main()
{
	vec3 worldPos=(invProjectionView*((gl_FragCoord-vec4(1.0))*2.0)).xyz;
	vec3 viewRay=worldPos-eyePos;
	float gamma=dot(out_normal,viewRay)/(sqrt(dot(out_normal,out_normal))*sqrt(dot(viewRay,viewRay)));

	alphaDensity=clamp(k/(area*gamma));
	alphaFade=clamp(1.0-out_alphaTime/maxTime);

	fragColor= vec4(color,alphaDensity*alphaShape*alphaCurvature*alphaFade);
}