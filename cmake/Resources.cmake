# =========================================================
# GRIM Resource Handling & Install Rules
# =========================================================

set(GRIM_RESOURCES
    nlp_rules.json
    synonyms.json
    ai_config.json
    memory.json
)

# Copy JSON resources post-build
foreach(FILE ${GRIM_RESOURCES})
  if(EXISTS "${CMAKE_SOURCE_DIR}/resources/${FILE}")
    set(SRC_FILE "${CMAKE_SOURCE_DIR}/resources/${FILE}")
  else()
    set(SRC_FILE "${CMAKE_BINARY_DIR}/${FILE}")
  endif()
  add_custom_command(TARGET GRIM POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${SRC_FILE}"
            "$<TARGET_FILE_DIR:GRIM>/${FILE}"
  )
endforeach()

# Font detection
set(GRIM_FONT_SRC "")
if(GRIM_FONT_PATH)
  set(GRIM_FONT_SRC "${GRIM_FONT_PATH}")
elseif(EXISTS "${CMAKE_SOURCE_DIR}/DejaVuSans.ttf")
  set(GRIM_FONT_SRC "${CMAKE_SOURCE_DIR}/DejaVuSans.ttf")
elseif(EXISTS "C:/Windows/Fonts/arial.ttf")
  set(GRIM_FONT_SRC "C:/Windows/Fonts/arial.ttf")
elseif(EXISTS "C:/Windows/Fonts/segoeui.ttf")
  set(GRIM_FONT_SRC "C:/Windows/Fonts/segoeui.ttf")
else()
  message(WARNING "No font found. Provide -DGRIM_FONT_PATH=...")
endif()

# Whisper model auto-download
set(WHISPER_MODEL_DIR "${CMAKE_SOURCE_DIR}/external/whisper.cpp/models")
set(WHISPER_MODEL_FILE "${WHISPER_MODEL_DIR}/ggml-small.bin")

if(NOT EXISTS "${WHISPER_MODEL_FILE}")
  file(MAKE_DIRECTORY "${WHISPER_MODEL_DIR}")
  add_custom_command(
    OUTPUT "${WHISPER_MODEL_FILE}"
    COMMAND ${CMAKE_COMMAND} -E echo "Downloading Whisper model (ggml-small.bin)..."
    COMMAND curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin
            -o "${WHISPER_MODEL_FILE}"
    COMMENT "Fetching Whisper model"
    VERBATIM
  )
  add_custom_target(fetch_whisper_model ALL DEPENDS "${WHISPER_MODEL_FILE}")
  add_dependencies(GRIM fetch_whisper_model)
endif()

# Copy resources folder
if(EXISTS "${CMAKE_SOURCE_DIR}/resources")
  add_custom_target(copy_resources ALL
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_SOURCE_DIR}/resources"
            "$<TARGET_FILE_DIR:GRIM>/resources"
  )
  add_dependencies(GRIM copy_resources)
endif()

# Install rules
include(GNUInstallDirs)
install(TARGETS GRIM RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

foreach(FILE ${GRIM_RESOURCES})
  if(EXISTS "${CMAKE_SOURCE_DIR}/resources/${FILE}")
    install(FILES "${CMAKE_SOURCE_DIR}/resources/${FILE}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
  else()
    install(FILES "${CMAKE_BINARY_DIR}/${FILE}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
  endif()
endforeach()

if(GRIM_FONT_SRC)
  install(FILES "${GRIM_FONT_SRC}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
endif()

if(EXISTS "${WHISPER_MODEL_FILE}")
  install(FILES "${WHISPER_MODEL_FILE}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim/models)
endif()
