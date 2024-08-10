#include "../core/scene.h"

#include <filesystem>

#define CGLTF_IMPLEMENTATION
#include <cgltf/cgltf.h>
#undef CGLTF_IMPLEMENTATION

#include "../core/debug_macros.h"

namespace
{
	void processGLTFNode(const cgltf_node* node, std::vector<SceneData::Instance>& instances, cgltf_mesh* firstMesh, size_t meshCount, const glm::mat4& transform)
	{
		cgltf_float localMat[16]{};
		cgltf_node_transform_local(node, localMat);
		glm::mat4 world{ transform * glm::mat4{
				glm::vec4{localMat[0], -localMat[1], -localMat[2], localMat[3]},
				glm::vec4{-localMat[4], localMat[5], localMat[6], localMat[7]},
				glm::vec4{-localMat[8], localMat[9], localMat[10], localMat[11]},
				glm::vec4{-localMat[12], localMat[13], localMat[14], localMat[15]}} };

		if (node->mesh != nullptr)
		{
			size_t index{ static_cast<size_t>(node->mesh - firstMesh) };
			R_ASSERT_LOG(index < meshCount, "Invalid mesh index from a pointer.");
			instances.emplace_back(static_cast<int>(index),
					glm::mat3x4{
					glm::vec4{world[0][0], world[1][0], world[2][0], world[3][0]},
					glm::vec4{world[0][1], world[1][1], world[2][1], world[3][1]},
					glm::vec4{world[0][2], world[1][2], world[2][2], world[3][2]} });
		}

		for (int i{ 0 }; i < node->children_count; ++i)
		{
			const cgltf_node* child{ node->children[i] };
			processGLTFNode(child, instances, firstMesh, meshCount, world);
		}
	}
	void processGLTFScene(const cgltf_scene* scene, std::vector<SceneData::Instance>& instances, cgltf_mesh* firstMesh, size_t meshCount, const glm::mat4& transform)
	{
		for (int i{ 0 }; i < scene->nodes_count; ++i)
			processGLTFNode(scene->nodes[i], instances, firstMesh, meshCount, transform);
	}

	SceneData::Model loadGLTF(const std::filesystem::path& path, std::vector<SceneData::MaterialDescriptor>& materialDescriptors, const SceneData::MaterialDescriptor* assignedDescriptor)
	{
		SceneData::Model model{};
		model.path = path;
		model.name = path.parent_path().filename().string();

		cgltf_options options{};
		cgltf_data* data{};
		cgltf_result result{};
		cgltf_mesh* firstMesh{};

		result = cgltf_parse_file(&options, path.string().c_str(), &data);
		R_ASSERT_LOG(result == cgltf_result_success, "Parsing GLTF file failed");

		cgltf_load_buffers(&options, data, path.string().c_str());

		R_ASSERT_LOG(data->meshes_count != 0, "Model doesn't contain any meshes");
		firstMesh = data->meshes;

		size_t meshCount{ data->meshes_count };
		for (int i{ 0 }; i < meshCount; ++i)
		{
			SceneData::Mesh& sceneMesh{ model.meshes.emplace_back() };

			const cgltf_mesh& mesh{ data->meshes[i] };
			for (int j{ 0 }; j < mesh.primitives_count; ++j)
			{
				SceneData::Submesh& sceneSubmesh{ sceneMesh.submeshes.emplace_back() };

				const cgltf_primitive& primitive{ mesh.primitives[j] };
				if (primitive.type != cgltf_primitive_type_triangles)
				{
					R_LOG("Unsupported primitive skipped");
					continue;
				}

				int matIndex{};
				if (assignedDescriptor == nullptr)
					R_ERR_LOG("Materials from GLTF are not supported yet.")
				else
				{
					matIndex = materialDescriptors.size();
					SceneData::MaterialDescriptor descriptor{ *assignedDescriptor };
					descriptor.name = descriptor.name + ' ' + '(' + mesh.name + ' ' + std::to_string(j) + ')';
					materialDescriptors.push_back(descriptor);
				}

				size_t count{ primitive.indices->count };
				size_t stride{ primitive.indices->stride };
				size_t offset{ primitive.indices->offset + primitive.indices->buffer_view->offset };

				R_ASSERT_LOG(primitive.attributes_count != 0, "No attributes in the mesh primitive");
				size_t vertCount{ primitive.attributes[0].data->count };
				switch (primitive.indices->component_type)
				{
					case cgltf_component_type_r_8u:
						{
							sceneSubmesh = SceneData::Submesh::createSubmesh(primitive.indices->count, IndexType::UINT_16, vertCount, matIndex);
							uint16_t* data{ reinterpret_cast<uint16_t*>(sceneSubmesh.indices) };
							R_ASSERT_LOG(stride == primitive.indices->buffer_view->stride || primitive.indices->buffer_view->stride == 0, "Buffer stride is not equal to accessor stride");
							for (int i{ 0 }; i < count; ++i)
								data[i] = static_cast<uint16_t>(reinterpret_cast<uint8_t*>(primitive.indices->buffer_view->buffer->data)[offset + i * stride]);
						}
						break;
					case cgltf_component_type_r_16u:
						{
							sceneSubmesh = SceneData::Submesh::createSubmesh(primitive.indices->count, IndexType::UINT_16, vertCount, matIndex);
							uint16_t* data{ reinterpret_cast<uint16_t*>(sceneSubmesh.indices) };
							R_ASSERT_LOG(stride == primitive.indices->buffer_view->stride || primitive.indices->buffer_view->stride == 0, "Buffer stride is not equal to accessor stride");
							for (int i{ 0 }; i < count; ++i)
								data[i] = *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(primitive.indices->buffer_view->buffer->data) + offset + i * stride);
						}
						break;
					case cgltf_component_type_r_32u:
						{
							sceneSubmesh = SceneData::Submesh::createSubmesh(primitive.indices->count, IndexType::UINT_32, vertCount, matIndex);
							uint32_t* data{ reinterpret_cast<uint32_t*>(sceneSubmesh.indices) };
							R_ASSERT_LOG(stride == primitive.indices->buffer_view->stride || primitive.indices->buffer_view->stride == 0, "Buffer stride is not equal to accessor stride");
							for (int i{ 0 }; i < count; ++i)
								data[i] = *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(primitive.indices->buffer_view->buffer->data) + offset + i * stride);
						}
						break;
					default:
						R_LOG("Unsupported index type");
						continue;
						break;
				}

				count = vertCount;
				for (int k{ 0 }; k < primitive.attributes_count; ++k)
				{
					const cgltf_attribute& attribute{ primitive.attributes[k] };

					stride = attribute.data->stride;
					R_ASSERT_LOG(stride == attribute.data->buffer_view->stride || attribute.data->buffer_view->stride == 0, "Buffer stride is not equal to accessor stride");
					offset = attribute.data->buffer_view->offset + attribute.data->offset;

					switch (attribute.type)
					{
						case cgltf_attribute_type_position:
							{
								for (size_t i{ 0 }; i < attribute.data->count; ++i)
								{
									uint8_t* bufferData{ reinterpret_cast<uint8_t*>(attribute.data->buffer_view->buffer->data) };
									float* vec{ reinterpret_cast<float*>(bufferData + offset + i * stride) };
									sceneSubmesh.vertices[i] = glm::vec4{-vec[0], vec[1], vec[2], 0.0f};
								}
							}
							break;
						case cgltf_attribute_type_normal:
							{
								for (size_t i{ 0 }; i < attribute.data->count; ++i)
								{
									uint8_t* bufferData{ reinterpret_cast<uint8_t*>(attribute.data->buffer_view->buffer->data) };
									float* vec{ reinterpret_cast<float*>(bufferData + offset + i * stride) };
									sceneSubmesh.normals[i] = glm::vec4{-vec[0], vec[1], vec[2], 0.0f};
								}
							}
							break;
						default:
							continue;
							break;
					}
				}
			}
		}

		glm::mat4 transform{ glm::identity<glm::mat4>() };
		transform[0] *= 150.0f;
		transform[1] *= 150.0f;
		transform[2] *= 150.0f;
		transform[3] += glm::vec4{-200.0f, 0.0f, 0.0f, 0.0f};

		processGLTFScene(data->scene, model.instances, firstMesh, meshCount, transform);

		cgltf_free(data);

		return model;
	}
}

void SceneData::loadModel(const std::filesystem::path& path, const MaterialDescriptor* assignedDescriptor)
{
	const std::string ext{ path.extension().string() };

	if (ext == ".gltf" || ext == ".glb")
		models.push_back(loadGLTF(path, materialDescriptors, assignedDescriptor));
	else
		R_LOG("Unknown model extension");
}
