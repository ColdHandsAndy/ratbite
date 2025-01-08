#include "../core/scene.h"

#include <fstream>
#include <filesystem>

#define CGLTF_IMPLEMENTATION
#include <cgltf/cgltf.h>
#undef CGLTF_IMPLEMENTATION
#include <stb/stb_image.h>

#include "../core/debug_macros.h"
#include "../core/sw_rast.h"

namespace
{
	void processGLTFNode(const cgltf_data* modelData, const cgltf_node* node,
			const glm::mat4& worldFromModel, glm::mat4 modelFromLocal,
			const std::vector<std::vector<SceneData::EmissiveMeshSubset>>& meshesEmissiveSubsets,
			size_t& triangleCount, std::vector<SceneData::Instance>& instances, std::vector<SceneData::EmissiveMeshSubset>& instancedEmissiveSubsets)
	{
		cgltf_float localMat[16]{};
		cgltf_node_transform_local(node, localMat);
		const cgltf_float* c0{ localMat + 0 };
		const cgltf_float* c1{ localMat + 4 };
		const cgltf_float* c2{ localMat + 8 };
		const cgltf_float* c3{ localMat + 12 };
		glm::mat4 coordinateCorrectedLocalTransform{
			glm::vec4{ c0[0],-c0[2],-c0[1],-c0[3]},
			glm::vec4{-c2[0], c2[2], c2[1], c2[3]},
			glm::vec4{-c1[0], c1[2], c1[1], c1[3]},
			glm::vec4{-c3[0], c3[2], c3[1], c3[3]},
		};
		modelFromLocal = modelFromLocal * coordinateCorrectedLocalTransform;

		if (node->mesh != nullptr)
		{
			size_t index{ cgltf_mesh_index(modelData, node->mesh) };

			instances.emplace_back(static_cast<int>(index),
				glm::mat4x3{
					glm::vec3{modelFromLocal[0][0], modelFromLocal[0][1], modelFromLocal[0][2]},
					glm::vec3{modelFromLocal[1][0], modelFromLocal[1][1], modelFromLocal[1][2]},
					glm::vec3{modelFromLocal[2][0], modelFromLocal[2][1], modelFromLocal[2][2]},
					glm::vec3{modelFromLocal[3][0], modelFromLocal[3][1], modelFromLocal[3][2]}, });

			for (int i{ 0 }; i < meshesEmissiveSubsets[index].size(); ++i)
			{
				auto& emissiveSubset{ meshesEmissiveSubsets[index][i] };
				if (emissiveSubset.triangles.size() != 0)
				{
					auto& instancedEmissiveSubset{ instancedEmissiveSubsets.emplace_back() };
					instancedEmissiveSubset.instanceIndex = instances.size() - 1;
					instancedEmissiveSubset.submeshIndex = emissiveSubset.submeshIndex;
					instancedEmissiveSubset.triangles = emissiveSubset.triangles;
					for (int i{ 0 }; i < instancedEmissiveSubset.triangles.size(); ++i)
					{
						instancedEmissiveSubset.triangles[i].v0WS = worldFromModel * modelFromLocal * glm::vec4{instancedEmissiveSubset.triangles[i].v0, 1.0f};
						instancedEmissiveSubset.triangles[i].v1WS = worldFromModel * modelFromLocal * glm::vec4{instancedEmissiveSubset.triangles[i].v1, 1.0f};
						instancedEmissiveSubset.triangles[i].v2WS = worldFromModel * modelFromLocal * glm::vec4{instancedEmissiveSubset.triangles[i].v2, 1.0f};
					}
					instancedEmissiveSubset.transformFluxCorrection = glm::abs(glm::determinant(glm::mat3{worldFromModel * modelFromLocal}));
				}
			}

			for (int i{ 0 }; i < node->mesh->primitives_count; ++i)
				if (node->mesh->primitives[i].attributes != nullptr)
					triangleCount += node->mesh->primitives[i].attributes[0].data->count;
		}

		for (int i{ 0 }; i < node->children_count; ++i)
		{
			const cgltf_node* child{ node->children[i] };
			processGLTFNode(modelData, child,
					worldFromModel, modelFromLocal,
					meshesEmissiveSubsets,
					triangleCount, instances, instancedEmissiveSubsets);
		}
	}
	void processGLTFSceneGraph(const cgltf_data* modelData,
			const std::vector<std::vector<SceneData::EmissiveMeshSubset>>& meshesEmissiveSubsets,
			const glm::mat4& worldFromModel,
			size_t& triangleCount,
			std::vector<SceneData::Instance>& instances, std::vector<SceneData::EmissiveMeshSubset>& instancedEmissiveSubsets)
	{
		for (int i{ 0 }; i < modelData->scene->nodes_count; ++i)
			processGLTFNode(modelData, modelData->scene->nodes[i],
					worldFromModel, glm::identity<glm::mat4>(),
					meshesEmissiveSubsets,
					triangleCount, instances, instancedEmissiveSubsets);
	}

	SceneData::Model loadGLTF(const std::filesystem::path& path, const glm::mat4& transform, uint32_t id)
	{
		SceneData::Model model{};
		model.id = id;
		model.path = path;
		model.name = path.filename().stem() == "scene" ? path.parent_path().filename().string() : path.stem().string();
		model.transform = transform;

		std::vector<SceneData::Model::ImageData>& imageData{ model.imageData };
		std::vector<SceneData::Model::TextureData>& textureData{ model.textureData };
		std::vector<SceneData::MaterialDescriptor>& materialDescriptors{ model.materialDescriptors };

		cgltf_options options{};
		cgltf_data* data{};
		cgltf_result result{};

		result = cgltf_parse_file(&options, path.string().c_str(), &data);
		R_ASSERT_LOG(result == cgltf_result_success, "Parsing GLTF file failed");
		R_ASSERT_LOG(data->meshes_count != 0, "Model doesn't contain any meshes");

		cgltf_load_buffers(&options, data, path.string().c_str());

		// Load Image and Texture data
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
			TextureAddress addressX{ TextureAddress::WRAP };
			TextureAddress addressY{ TextureAddress::WRAP };
			if (tex->sampler != nullptr)
			{
				if (tex->sampler->mag_filter == 9728)
					filter = TextureFilter::NEAREST;
				else if (tex->sampler->mag_filter == 9729)
					filter = TextureFilter::LINEAR;

				if (tex->sampler->wrap_s == 10497)
					addressX = TextureAddress::WRAP;
				else if (tex->sampler->wrap_s == 33071)
					addressX = TextureAddress::CLAMP;
				else if (tex->sampler->wrap_s == 33648)
					addressX = TextureAddress::MIRROR;

				if (tex->sampler->wrap_t == 10497)
					addressY = TextureAddress::WRAP;
				else if (tex->sampler->wrap_t == 33071)
					addressY = TextureAddress::CLAMP;
				else if (tex->sampler->wrap_t == 33648)
					addressY = TextureAddress::MIRROR;
			}

			textureData.emplace_back(static_cast<int>(cgltf_image_index(data, tex->image)), false, filter, addressX, addressY);
		}

		// Create emissive subsets for every mesh but only fill ones with emissive materials
		std::vector<std::vector<SceneData::EmissiveMeshSubset>> meshesEmissiveSubsets(data->meshes_count);

		// Process meshes and submeshes
		for (int i{ 0 }; i < data->meshes_count; ++i)
		{
			SceneData::Mesh& sceneMesh{ model.meshes.emplace_back() };

			const cgltf_mesh& mesh{ data->meshes[i] };
			for (int j{ 0 }; j < mesh.primitives_count; ++j)
			{
				SceneData::Submesh& sceneSubmesh{ sceneMesh.submeshes.emplace_back() };
				bool emissiveFactorPresent{ false };
				bool emissiveTexturePresent{ false };

				const cgltf_primitive& primitive{ mesh.primitives[j] };
				if (primitive.type != cgltf_primitive_type_triangles)
				{
					R_LOG("Unsupported primitive skipped");
					continue;
				}

				// Create material descriptor from material data
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
							if (material->alpha_mode == cgltf_alpha_mode_mask)
								descriptor.alphaInterpretation = SceneData::MaterialDescriptor::AlphaInterpretation::CUTOFF;
							else if (material->alpha_mode == cgltf_alpha_mode_blend)
								descriptor.alphaInterpretation = SceneData::MaterialDescriptor::AlphaInterpretation::BLEND;
							else
								descriptor.alphaInterpretation = SceneData::MaterialDescriptor::AlphaInterpretation::NONE;
							descriptor.alphaCutoff = material->alpha_cutoff;
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

					if (material->has_sheen &&
						!(material->sheen.sheen_color_factor[0] == 0.0f &&
						  material->sheen.sheen_color_factor[1] == 0.0f &&
						  material->sheen.sheen_color_factor[2] == 0.0f))
					{
						descriptor.sheenPresent = true;
						if (material->sheen.sheen_color_factor[0] != 1.0f ||
							material->sheen.sheen_color_factor[1] != 1.0f ||
							material->sheen.sheen_color_factor[2] != 1.0f)
						{
							descriptor.sheenColorFactorPresent = true;
							descriptor.sheenColorFactor[0] =
								material->sheen.sheen_color_factor[0];
							descriptor.sheenColorFactor[1] =
								material->sheen.sheen_color_factor[1];
							descriptor.sheenColorFactor[2] =
								material->sheen.sheen_color_factor[2];
						}
						descriptor.sheenRoughnessFactorPresent = true;
						descriptor.sheenRoughnessFactor =
							material->sheen.sheen_roughness_factor;

						if (material->sheen.sheen_color_texture.texture != nullptr)
						{
							descriptor.sheenColorTextureIndex = cgltf_texture_index(data, material->sheen.sheen_color_texture.texture);
							descriptor.shcTexCoordIndex = material->sheen.sheen_color_texture.texcoord;
						}
						if (material->sheen.sheen_roughness_texture.texture != nullptr)
						{
							descriptor.sheenRoughTextureIndex = cgltf_texture_index(data, material->sheen.sheen_roughness_texture.texture);
							descriptor.shrTexCoordIndex = material->sheen.sheen_roughness_texture.texcoord;
						}
					}

					if (!(material->emissive_factor[0] == 0.0f &&
						  material->emissive_factor[1] == 0.0f &&
						  material->emissive_factor[2] == 0.0f))
					{
						emissiveFactorPresent = true;
						descriptor.emissiveFactorPresent = true;

						const float* emissiveFactor{ material->emissive_factor };
						if (material->has_emissive_strength)
						{
							descriptor.emissiveFactor[0] = emissiveFactor[0] * material->emissive_strength.emissive_strength;
							descriptor.emissiveFactor[1] = emissiveFactor[1] * material->emissive_strength.emissive_strength;
							descriptor.emissiveFactor[2] = emissiveFactor[2] * material->emissive_strength.emissive_strength;
						}
						else
						{
							descriptor.emissiveFactor[0] = emissiveFactor[0];
							descriptor.emissiveFactor[1] = emissiveFactor[1];
							descriptor.emissiveFactor[2] = emissiveFactor[2];
						}

						if (material->emissive_texture.texture != nullptr)
						{
							emissiveTexturePresent = true;
							descriptor.emissiveTextureIndex = cgltf_texture_index(data, material->emissive_texture.texture);
							descriptor.emTexCoordIndex = material->emissive_texture.texcoord;
						}
					}
				}
				materialDescriptors.push_back(descriptor);


				// Load indices for a submesh
				{
					R_ASSERT_LOG(primitive.attributes_count != 0, "No attributes in the mesh primitive");
					size_t vertCount{ primitive.attributes[0].data->count };
					size_t stride{ primitive.indices->stride };
					size_t offset{ primitive.indices->offset + primitive.indices->buffer_view->offset };
					switch (primitive.indices->component_type)
					{
						case cgltf_component_type_r_8u:
							{
								sceneSubmesh = SceneData::Submesh::createSubmesh(primitive.indices->count, IndexType::UINT_16, vertCount, matIndex);
								uint16_t* data{ reinterpret_cast<uint16_t*>(sceneSubmesh.indices) };
								R_ASSERT_LOG(stride == primitive.indices->buffer_view->stride || primitive.indices->buffer_view->stride == 0, "Buffer stride is not equal to accessor stride");
								for (int k{ 0 }; k < primitive.indices->count; ++k)
									data[k] = static_cast<uint16_t>(reinterpret_cast<uint8_t*>(primitive.indices->buffer_view->buffer->data)[offset + k * stride]);
							}
							break;
						case cgltf_component_type_r_16u:
							{
								sceneSubmesh = SceneData::Submesh::createSubmesh(primitive.indices->count, IndexType::UINT_16, vertCount, matIndex);
								uint16_t* data{ reinterpret_cast<uint16_t*>(sceneSubmesh.indices) };
								R_ASSERT_LOG(stride == primitive.indices->buffer_view->stride || primitive.indices->buffer_view->stride == 0, "Buffer stride is not equal to accessor stride");
								for (int k{ 0 }; k < primitive.indices->count; ++k)
									data[k] = *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(primitive.indices->buffer_view->buffer->data) + offset + k * stride);
							}
							break;
						case cgltf_component_type_r_32u:
							{
								sceneSubmesh = SceneData::Submesh::createSubmesh(primitive.indices->count, IndexType::UINT_32, vertCount, matIndex);
								uint32_t* data{ reinterpret_cast<uint32_t*>(sceneSubmesh.indices) };
								R_ASSERT_LOG(stride == primitive.indices->buffer_view->stride || primitive.indices->buffer_view->stride == 0, "Buffer stride is not equal to accessor stride");
								for (int k{ 0 }; k < primitive.indices->count; ++k)
									data[k] = *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(primitive.indices->buffer_view->buffer->data) + offset + k * stride);
							}
							break;
						default:
							R_LOG("Unsupported index type");
							continue;
							break;
					}
				}

				// Load vertex attributes for a submesh
				for (int k{ 0 }; k < primitive.attributes_count; ++k)
				{
					const cgltf_attribute& attribute{ primitive.attributes[k] };

					size_t stride{ attribute.data->stride };
					R_ASSERT_LOG(stride == attribute.data->buffer_view->stride || attribute.data->buffer_view->stride == 0, "Buffer stride is not equal to accessor stride");
					size_t offset{ attribute.data->buffer_view->offset + attribute.data->offset };

					auto coordinateChange{ [](const glm::vec3& v)->glm::vec3 { return {-v.x, v.z, v.y}; } };
					switch (attribute.type)
					{
						case cgltf_attribute_type_position:
							{
								for (size_t i{ 0 }; i < attribute.data->count; ++i)
								{
									uint8_t* bufferData{ reinterpret_cast<uint8_t*>(attribute.data->buffer_view->buffer->data) };
									float* vec{ reinterpret_cast<float*>(bufferData + offset + i * stride) };
									sceneSubmesh.vertices[i] = glm::vec4{coordinateChange({vec[0], vec[1], vec[2]}), 0.0f};
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
									sceneSubmesh.normals[i] = glm::vec4{coordinateChange({vec[0], vec[1], vec[2]}), 0.0f};
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
									sceneSubmesh.tangents[i] = glm::vec4{coordinateChange({vec[0], vec[1], vec[2]}), vec[3]};
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

				// Cut emissive triangles from a submesh
				if (emissiveFactorPresent || emissiveTexturePresent)
				{
					auto& emissiveSubset{ meshesEmissiveSubsets[i].emplace_back() };
					emissiveSubset.submeshIndex = j;

					uint32_t triangleCount{ sceneSubmesh.primitiveCount };
					emissiveSubset.triangles.reserve(triangleCount);
					for (uint32_t k{ 0 }; k < triangleCount; ++k)
					{
						// Get indices so we can get primitive attributes
						uint32_t indices[3]{};
						switch (sceneSubmesh.indexType)
						{
							case IndexType::UINT_16:
								{
									indices[0] = *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(sceneSubmesh.indices) + (k * 3 + 0) * sizeof(uint16_t));
									indices[1] = *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(sceneSubmesh.indices) + (k * 3 + 1) * sizeof(uint16_t));
									indices[2] = *reinterpret_cast<uint16_t*>(reinterpret_cast<uint8_t*>(sceneSubmesh.indices) + (k * 3 + 2) * sizeof(uint16_t));
								}
								break;
							case IndexType::UINT_32:
								{
									indices[0] = *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(sceneSubmesh.indices) + (k * 3 + 0) * sizeof(uint32_t));
									indices[1] = *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(sceneSubmesh.indices) + (k * 3 + 1) * sizeof(uint32_t));
									indices[2] = *reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(sceneSubmesh.indices) + (k * 3 + 2) * sizeof(uint32_t));
								}
								break;
							default:
								R_LOG("Unsupported index type");
								continue;
						}

						// Triangle flux initialized by the factor
						float triangleFlux{
							(descriptor.emissiveFactor[0] +
							descriptor.emissiveFactor[1] +
							descriptor.emissiveFactor[2])
							/ 3.0f };
						// If the emission is texture based we calculate it separately
						glm::vec2 uvs[3]{};
						if (emissiveTexturePresent)
						{
							const auto& tex{ model.textureData[descriptor.emissiveTextureIndex] };
							const auto& im{ model.imageData[tex.imageIndex] };
							TextureAddress texAddressingX{ tex.addressX };
							TextureAddress texAddressingY{ tex.addressY };
							const uint32_t texWidth{ im.width };
							const uint32_t texHeigth{ im.height };
							uvs[0] = sceneSubmesh.texCoordsSets[descriptor.emTexCoordIndex][indices[0]];
							uvs[1] = sceneSubmesh.texCoordsSets[descriptor.emTexCoordIndex][indices[1]];
							uvs[2] = sceneSubmesh.texCoordsSets[descriptor.emTexCoordIndex][indices[2]];
							const glm::vec2 texuv[3]{ uvs[0] * glm::vec2{texWidth, texHeigth},
								uvs[1] * glm::vec2{texWidth, texHeigth},
								uvs[2] * glm::vec2{texWidth, texHeigth} };
							double averageEmission{ 0.0 };
							uint32_t texelsFetched{ 0 };
							// This function gathers the average emission from a texel
							const std::function<void(int, int)> computeTexFlux{ [&tex, &im, &averageEmission, &texelsFetched](int x, int y)
								{
									if (tex.addressX == TextureAddress::WRAP)
									{
										x = (x >= 0) ? x : x + im.width * ((-x - 1) / im.width + 1);
										x = x % im.width;
									}
									else if (tex.addressX == TextureAddress::CLAMP)
									{
										x = min(max(0, x), static_cast<int>(im.width) - 1);
									}
									else if (tex.addressX == TextureAddress::MIRROR)
									{
										x = (x >= 0) ? x : x + im.width * ((-x - 1) / im.width + 2);
										bool even{ ((x / im.width) % 2) == 0 };
										x = x % im.width;
										if (even)
											x = im.width - 1 - x;
									}
									else
										x = 0;

									if (tex.addressY == TextureAddress::WRAP)
									{
										y = (y >= 0) ? y : y + im.height * ((-y - 1) / im.height + 1);
										y = y % im.height;
									}
									else if (tex.addressY == TextureAddress::CLAMP)
									{
										y = min(max(0, y), static_cast<int>(im.height) - 1);
									}
									else if (tex.addressY == TextureAddress::MIRROR)
									{
										y = (y >= 0) ? y : y + im.height * ((-y - 1) / im.height + 2);
										bool even{ ((y / im.height) % 2) == 0 };
										y = y % im.height;
										if (even)
											y = im.height - 1 - y;
									}
									else
										y = 0;

									float r{ reinterpret_cast<uint8_t*>(im.data)[4 * (im.width * y + x) + 0] / 255.0f };
									float g{ reinterpret_cast<uint8_t*>(im.data)[4 * (im.width * y + x) + 1] / 255.0f };
									float b{ reinterpret_cast<uint8_t*>(im.data)[4 * (im.width * y + x) + 2] / 255.0f };
									float a{ reinterpret_cast<uint8_t*>(im.data)[4 * (im.width * y + x) + 3] / 255.0f };

									averageEmission += (r + g + b) * a / 3.0f;
									++texelsFetched;
								} };
							// Rasterization
							SoftwareRasterization::drawTriangle2D(&(texuv[0][0]), &(texuv[1][0]), &(texuv[2][0]), computeTexFlux);
							// If no texels are sampled just use an average from each uv vertex
							if (texelsFetched != 0)
								triangleFlux *= averageEmission / texelsFetched;
							else
							{
								averageEmission = 0.0;
								texelsFetched = 0;
								for (int a{ 0 }; a < 3; ++a)
								{
									computeTexFlux(texuv[a].x, texuv[a].y);
								}
								triangleFlux *= averageEmission / texelsFetched;
							}
						}
						// Adjust the flux by primitive area and diffuse distribution (Pi)
						glm::vec3 vertices[3]{
							sceneSubmesh.vertices[indices[0]],
							sceneSubmesh.vertices[indices[1]],
							sceneSubmesh.vertices[indices[2]], };
						triangleFlux *= glm::length(glm::cross(vertices[1] - vertices[0], vertices[2] - vertices[0])) * 0.5f * glm::pi<float>();
						// Cull if the flux is negligible
						if (triangleFlux > 0.0f)
						{
							// Make emissive triangles degenerate since we need to create separate AC for them
							sceneSubmesh.discardedPrimitives.push_back(k);

							emissiveSubset.triangles.push_back(SceneData::EmissiveMeshSubset::TriangleData{
									.v0 = vertices[0],
									.v1 = vertices[1],
									.v2 = vertices[2],
									.uv0 = uvs[0],
									.uv1 = uvs[1],
									.uv2 = uvs[2],
									.primIndex = k,
									.flux = triangleFlux });
						}
					}
				}
			}
		}

		processGLTFSceneGraph(data, meshesEmissiveSubsets, model.transform, model.triangleCount, model.instances, model.instancedEmissiveMeshSubsets);

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
