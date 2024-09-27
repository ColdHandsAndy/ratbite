#include "../core/scene.h"

#include <fstream>
#include <filesystem>

#define CGLTF_IMPLEMENTATION
#include <cgltf/cgltf.h>
#undef CGLTF_IMPLEMENTATION
#include <stb/stb_image.h>

#include "../core/debug_macros.h"

namespace
{
	void processGLTFNode(const cgltf_data* modelData, const cgltf_node* node, size_t& triangleCount, std::vector<SceneData::Instance>& instances, cgltf_mesh* firstMesh, size_t meshCount, const glm::mat4& transform)
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
			size_t index{ cgltf_mesh_index(modelData, node->mesh) };
			instances.emplace_back(static_cast<int>(index),
				glm::mat4x3{
					glm::vec3{world[0][0], world[0][1], world[0][2]},
					glm::vec3{world[1][0], world[1][1], world[1][2]},
					glm::vec3{world[2][0], world[2][1], world[2][2]},
					glm::vec3{world[3][0], world[3][1], world[3][2]}, });

			for (int i{ 0 }; i < node->mesh->primitives_count; ++i)
				if (node->mesh->primitives[i].attributes != nullptr)
					triangleCount += node->mesh->primitives[i].attributes[0].data->count;
		}

		for (int i{ 0 }; i < node->children_count; ++i)
		{
			const cgltf_node* child{ node->children[i] };
			processGLTFNode(modelData, child, triangleCount, instances, firstMesh, meshCount, world);
		}
	}
	void processGLTFScene(const cgltf_data* modelData, const cgltf_scene* scene, size_t& triangleCount, std::vector<SceneData::Instance>& instances, cgltf_mesh* firstMesh, size_t meshCount)
	{
		for (int i{ 0 }; i < scene->nodes_count; ++i)
			processGLTFNode(modelData, scene->nodes[i], triangleCount, instances, firstMesh, meshCount, glm::identity<glm::mat4>());
	}

	SceneData::Model loadGLTF(const std::filesystem::path& path, const glm::mat4& transform, uint32_t id)
	{
		SceneData::Model model{};
		model.id = id;
		model.path = path;
		model.name = path.parent_path().filename().string();
		model.transform = transform;

		std::vector<SceneData::Model::ImageData>& imageData{ model.imageData };
		std::vector<SceneData::Model::TextureData>& textureData{ model.textureData };
		std::vector<SceneData::MaterialDescriptor>& materialDescriptors{ model.materialDescriptors };

		cgltf_options options{};
		cgltf_data* data{};
		cgltf_result result{};
		cgltf_mesh* firstMesh{};

		result = cgltf_parse_file(&options, path.string().c_str(), &data);
		R_ASSERT_LOG(result == cgltf_result_success, "Parsing GLTF file failed");
		R_ASSERT_LOG(data->meshes_count != 0, "Model doesn't contain any meshes");

		cgltf_load_buffers(&options, data, path.string().c_str());

		auto loadImage( [&](const char* uri, const cgltf_buffer_view* bufferView) -> int{
				int index{};
				if (uri)
				{
					if (strncmp(uri, "data:", 5) == 0)
					{
						// URI is "data:"
						// const char* comma{ strchr(uri, ',') };
						// if (comma && comma - uri >= 7 && strncmp(comma - 7, ";base64", 7) == 0)
						// 	cgltf_load_buffer_base64(&options, /*Image byte size*/, comma + 1, &data);
						R_ERR_LOG("GLTF image data is base64 encoded. Not supported yet");
					}
					else
					{
						// URI is path
						char* pathToTex{ new char[path.string().length() + strlen(uri) + 1] };
						cgltf_combine_paths(pathToTex, path.string().c_str(), uri);
						cgltf_decode_uri(pathToTex + strlen(pathToTex) - strlen(uri));

						int x{};
						int y{};
						int c{};
						stbi_info(pathToTex, &x, &y, &c);
						int cNum{ c == 1 ? c : 4 };
						unsigned char* image{ stbi_load(pathToTex, &x, &y, &c, cNum) };

						size_t byteSize{ static_cast<size_t>(x * y * cNum) };
						void* data{ malloc(byteSize) };
						memcpy(data, image, byteSize);
						index = imageData.size();
						imageData.emplace_back(data, x, y, cNum, byteSize);

						stbi_image_free(image);

						delete[] pathToTex;
					}
				}
				else
				{
					int x{};
					int y{};
					int c{};
					stbi_info_from_memory(reinterpret_cast<unsigned char*>(bufferView->buffer->data) + bufferView->offset, bufferView->size, &x, &y, &c);
					int cNum{ c == 1 ? c : 4 };
					unsigned char* image{ stbi_load_from_memory(reinterpret_cast<unsigned char*>(bufferView->buffer->data) + bufferView->offset, bufferView->size, &x, &y, &c, cNum) };

					size_t byteSize{ static_cast<size_t>(x * y * cNum) };
					void* data{ malloc(byteSize) };
					memcpy(data, image, byteSize);
					index = imageData.size();
					imageData.emplace_back(data, x, y, cNum, byteSize);

					stbi_image_free(image);
				}
				return index;
			} );
		for (int i{ 0 }; i < data->images_count; ++i)
			loadImage(data->images[i].uri, data->images[i].buffer_view);
		for (int i{ 0 }; i < data->textures_count; ++i)
		{
			const cgltf_texture* tex{ data->textures + i };
			TextureFilter filter{ TextureFilter::LINEAR };
			if (tex->sampler->mag_filter == 9728)
				filter = TextureFilter::NEAREST;
			else if (tex->sampler->mag_filter == 9729)
				filter = TextureFilter::LINEAR;

			TextureAddress addressX{};
			if (tex->sampler->wrap_s == 10497)
				addressX = TextureAddress::WRAP;
			else if (tex->sampler->wrap_s == 33071)
				addressX = TextureAddress::CLAMP;
			else if (tex->sampler->wrap_s == 33648)
				addressX = TextureAddress::MIRROR;
			else
				addressX = TextureAddress::WRAP;

			TextureAddress addressY{};
			if (tex->sampler->wrap_t == 10497)
				addressY = TextureAddress::WRAP;
			else if (tex->sampler->wrap_t == 33071)
				addressY = TextureAddress::CLAMP;
			else if (tex->sampler->wrap_t == 33648)
				addressY = TextureAddress::MIRROR;
			else
				addressY = TextureAddress::WRAP;

			textureData.emplace_back(static_cast<int>(cgltf_image_index(data, tex->image)), false, filter, addressX, addressY);
		}

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

				const cgltf_material* material{ primitive.material };
				int matIndex{ static_cast<int>(materialDescriptors.size()) };
				SceneData::MaterialDescriptor descriptor{};
				descriptor.bxdf = SceneData::BxDF::COMPLEX_SURFACE;
				descriptor.name = descriptor.name + ' ' + '(' + mesh.name + ' ' + std::to_string(j) + ')';
				if (material != nullptr)
				{
					descriptor.doubleSided = material->double_sided;
					descriptor.ior = material->has_ior ? material->ior.ior : 1.5f;
					if (material->has_pbr_metallic_roughness)
					{
						if (material->pbr_metallic_roughness.base_color_texture.texture != nullptr)
						{
							textureData[cgltf_texture_index(data, material->pbr_metallic_roughness.base_color_texture.texture)].sRGB = true;
							descriptor.baseColorTextureIndex = cgltf_texture_index(data, material->pbr_metallic_roughness.base_color_texture.texture);
							descriptor.bcTexCoordIndex = material->pbr_metallic_roughness.base_color_texture.texcoord;
							if (material->alpha_mode != cgltf_alpha_mode_opaque)
							{
								descriptor.alphaCutoffPresent = true;
								descriptor.alphaCutoff = material->alpha_cutoff;
							}
						}
						if (material->pbr_metallic_roughness.base_color_factor[0] != 1.0f ||
								material->pbr_metallic_roughness.base_color_factor[1] != 1.0f ||
								material->pbr_metallic_roughness.base_color_factor[2] != 1.0f ||
								material->pbr_metallic_roughness.base_color_factor[3] != 1.0f)
						{
							descriptor.bcFactorPresent = true;
							descriptor.baseColorFactor[0] = material->pbr_metallic_roughness.base_color_factor[0];
							descriptor.baseColorFactor[1] = material->pbr_metallic_roughness.base_color_factor[1];
							descriptor.baseColorFactor[2] = material->pbr_metallic_roughness.base_color_factor[2];
							descriptor.baseColorFactor[3] = material->pbr_metallic_roughness.base_color_factor[3];
						}
						if (material->pbr_metallic_roughness.metallic_factor != 1.0f)
						{
							descriptor.metFactorPresent = true;
							descriptor.metalnessFactor = material->pbr_metallic_roughness.metallic_factor;
						}
						if (material->pbr_metallic_roughness.roughness_factor != 1.0f)
						{
							descriptor.roughFactorPresent = true;
							descriptor.roughnessFactor = material->pbr_metallic_roughness.roughness_factor;
						}

						if (material->pbr_metallic_roughness.metallic_roughness_texture.texture != nullptr)
						{
							descriptor.metalRoughnessTextureIndex = cgltf_texture_index(data, material->pbr_metallic_roughness.metallic_roughness_texture.texture);
							descriptor.mrTexCoordIndex = material->pbr_metallic_roughness.metallic_roughness_texture.texcoord;
						}
					}
					if (material->has_transmission)
					{
						if (material->transmission.transmission_texture.texture != nullptr)
						{
							descriptor.transmissionTextureIndex = cgltf_texture_index(data, material->transmission.transmission_texture.texture);
							descriptor.trTexCoordIndex = material->transmission.transmission_texture.texcoord;
						}
						if (material->transmission.transmission_factor != 0.0f)
						{
							descriptor.transmitFactorPresent = true;
							descriptor.transmitFactor = material->transmission.transmission_factor;
						}
					}
					if (material->normal_texture.texture != nullptr)
					{
						descriptor.normalTextureIndex = cgltf_texture_index(data, material->normal_texture.texture);
						descriptor.nmTexCoordIndex = material->normal_texture.texcoord;
					}
				}
				materialDescriptors.push_back(descriptor);

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
								sceneSubmesh.addNormals();
								for (size_t i{ 0 }; i < attribute.data->count; ++i)
								{
									uint8_t* bufferData{ reinterpret_cast<uint8_t*>(attribute.data->buffer_view->buffer->data) };
									float* vec{ reinterpret_cast<float*>(bufferData + offset + i * stride) };
									sceneSubmesh.normals[i] = glm::vec4{-vec[0], vec[1], vec[2], 0.0f};
								}
							}
							break;
						case cgltf_attribute_type_tangent:
							{
								sceneSubmesh.addTangents();
								for (size_t i{ 0 }; i < attribute.data->count; ++i)
								{
									uint8_t* bufferData{ reinterpret_cast<uint8_t*>(attribute.data->buffer_view->buffer->data) };
									float* vec{ reinterpret_cast<float*>(bufferData + offset + i * stride) };
									sceneSubmesh.tangents[i] = glm::vec4{-vec[0], vec[1], vec[2], vec[3]};
								}
							}
							break;
						case cgltf_attribute_type_texcoord:
							{
								sceneSubmesh.addTexCoordsSet(attribute.index);
								for (size_t i{ 0 }; i < attribute.data->count; ++i)
								{
									uint8_t* bufferData{ reinterpret_cast<uint8_t*>(attribute.data->buffer_view->buffer->data) };
									glm::vec2 vec{};
									switch (attribute.data->component_type)
									{
										case cgltf_component_type_r_8u:
											bufferData = bufferData + offset + i * stride;
											vec = glm::vec2{reinterpret_cast<uint8_t*>(bufferData)[0] * (1.0f / 255.0f), reinterpret_cast<uint8_t*>(bufferData)[1] * (1.0f / 255.0f)};
											break;
										case cgltf_component_type_r_16u:
											bufferData = bufferData + offset + i * stride;
											vec = glm::vec2{reinterpret_cast<uint16_t*>(bufferData)[0] * (1.0f / 65535.0f), reinterpret_cast<uint16_t*>(bufferData)[1] * (1.0f / 65535.0f)};
											break;
										case cgltf_component_type_r_32f:
											bufferData = bufferData + offset + i * stride;
											vec = glm::vec2{reinterpret_cast<float*>(bufferData)[0], reinterpret_cast<float*>(bufferData)[1]};
											break;
										default:
											R_ERR_LOG("Unsupported component type");
											break;
									}
									sceneSubmesh.texCoordsSets[attribute.index][i] = vec;
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

		processGLTFScene(data, data->scene, model.triangleCount, model.instances, firstMesh, meshCount);

		cgltf_free(data);

		return model;
	}
}

int SceneData::loadModel(const std::filesystem::path& path, const glm::mat4& transform)
{
	const std::string ext{ path.extension().string() };

	int index{ static_cast<int>(models.size()) };

	if (ext == ".gltf" || ext == ".glb")
		models.push_back(loadGLTF(path, transform, ++m_idGen));
	else
		R_LOG("Unknown model extension");

	return index;
}
