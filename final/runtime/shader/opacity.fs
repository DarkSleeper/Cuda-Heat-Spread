out vec4 FragColor;

in vec2 TexCoords;
in vec3 world_normal;
in vec4 obj_color;

//uniform sampler2D texture_diffuse1;
uniform vec3 direct_light;

void main()
{
    //vec3 direct_light=vec3(0.0,1.0,0.0);
    vec3 light_dir=normalize(-1.0*direct_light);
    float diffuse_factor=clamp(dot(light_dir,world_normal),0.0,1.0);

    //vec4 color = texture(texture_diffuse1, TexCoords);
    vec4 color = obj_color;

    //default
    FragColor = vec4(vec3(color*0.5+0.5*color*diffuse_factor),1.0);

}