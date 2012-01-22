#version 330

uniform mat4 invProjectionView;
uniform vec3 eyePos;
uniform float k;//constant: height_of_the_prism*number_of_particles*constant
uniform vec3 fragColor;
uniform vec2 viewPort;

in vec4 gs_out_worldPos;
in vec3 gs_out_normal;
in float gs_out_alphaTime;
in float gs_out_alphaCurvature;
in float gs_out_alphaShape;
in float gs_out_area;

out vec4 fs_out_Color;

void main()
{
	//vec2 ndc = (gl_FragCoord.xy/viewPort.xy-0.5)*2.0;
	//vec4 worldPos=vec4(ndc,gl_FragCoord.z,1.0)/gl_FragCoord.w;
	//worldPos=invProjectionView*worldPos;

	vec4 tmp=vec4(gs_out_worldPos.xyz,1);
	//vec4 tmp=gs_out_worldPos;
	//tmp=invProjectionView*vec4(tmp.xyzw);
	//tmp.xyz/tmp.w;

	vec3 viewRay=normalize(gs_out_worldPos.xyz-eyePos);

	float gamma=dot(gs_out_normal,viewRay);///(sqrt(dot(gs_out_normal,gs_out_normal))*sqrt(dot(viewRay,viewRay)));

	float alphaDensity=clamp(k/(gs_out_area*gamma),0.0,1.0);
	float alphaFade=clamp(1.0-gs_out_alphaTime,0.0,1.0);

	fs_out_Color=vec4(fragColor,alphaDensity*alphaFade*gs_out_alphaShape/*alphaDensity*alphaFade*gs_out_alphaShape*gs_out_alphaCurvature*/);
}