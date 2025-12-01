import bpy
import json
import os

# Import the GLB file
glb_path = os.path.join(os.path.dirname(__file__), "character_2_5d.glb")
bpy.ops.import_scene.gltf(filepath=glb_path)

# Get the imported mesh
mesh_obj = bpy.context.selected_objects[0]
mesh = mesh_obj.data

# Read armature data from metadata (stored during export)
if 'bones' in mesh:
    # Create armature
    armature_data = bpy.data.armatures.new('CharacterArmature')
    armature_obj = bpy.data.objects.new('CharacterArmature', armature_data)
    bpy.context.collection.objects.link(armature_obj)
    
    # Enter edit mode to create bones
    bpy.context.view_layer.objects.active = armature_obj
    bpy.ops.object.mode_set(mode='EDIT')
    
    bones_data = json.loads(mesh['bones'])
    created_bones = {}
    
    # Create all bones
    for bone_data in bones_data:
        bone = armature_data.edit_bones.new(bone_data['name'])
        bone.head = tuple(bone_data['head'])
        bone.tail = tuple(bone_data['tail'])
        created_bones[bone_data['name']] = bone
    
    # Set parent relationships
    for bone_data in bones_data:
        if bone_data['parent']:
            created_bones[bone_data['name']].parent = created_bones[bone_data['parent']]
    
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Parent mesh to armature
    mesh_obj.parent = armature_obj
    modifier = mesh_obj.modifiers.new(name='Armature', type='ARMATURE')
    modifier.object = armature_obj
    
    # Apply vertex weights if available
    if 'vertex_weights' in mesh:
        weights = json.loads(mesh['vertex_weights'])
        for bone_idx, bone_data in enumerate(bones_data):
            vgroup = mesh_obj.vertex_groups.new(name=bone_data['name'])
            for v_idx, weight in enumerate(weights):
                if weight[bone_idx] > 0.01:  # Only add significant weights
                    vgroup.add([v_idx], weight[bone_idx], 'REPLACE')
    
    print("✓ Armature created and mesh rigged!")
    print("✓ You can now use Rigify to generate a full control rig:")
    print("  1. Select the armature")
    print("  2. Go to Armature Properties")
    print("  3. Enable 'Rigify' addon if not already enabled")
    print("  4. Click 'Generate Rig'")

else:
    print("No armature data found in GLB metadata")
