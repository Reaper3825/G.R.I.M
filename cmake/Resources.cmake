# =========================================================
# GRIM Resource Handling & Install Rules  (idempotent)
# =========================================================
include_guard(GLOBAL)

set(GRIM_RESOURCES
    nlp_rules.json
    synonyms.json
    ai_config.json
    memory.json
)

# Optional assets (safe if missing)
set(GRIM_FONT_SRC "${CMAKE_SOURCE_DIR}/resources/fonts/Inter/Inter-Regular.ttf" CACHE FILEPATH "Fallback UI font")
set(WHISPER_MODEL_FILE "${CMAKE_SOURCE_DIR}/models/ggml-base.en.bin" CACHE FILEPATH "Default Whisper model path")

# ---------- helper: copy a single file ----------
function(_grim_copy_one SRC DEST)
    add_custom_command(
        OUTPUT "${DEST}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "$<IF:$<BOOL:${DEST}>,${CMAKE_BINARY_DIR},${CMAKE_BINARY_DIR}>"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SRC}" "${DEST}"
        DEPENDS "${SRC}"
        VERBATIM
    )
endfunction()

# ---------- create a single, reusable copy target ----------
if(NOT TARGET grim_copy_resources)
    add_custom_target(grim_copy_resources ALL COMMENT "[GRIM] Copying runtime resources")

    # JSON resources → build dir
    foreach(FILE ${GRIM_RESOURCES})
        if(EXISTS "${CMAKE_SOURCE_DIR}/resources/${FILE}")
            set(SRC_FILE "${CMAKE_SOURCE_DIR}/resources/${FILE}")
        else()
            # allow user to place overrides next to the tree
            set(SRC_FILE "${CMAKE_BINARY_DIR}/${FILE}")
        endif()

        set(DST_FILE "${CMAKE_BINARY_DIR}/${FILE}")

        if(EXISTS "${SRC_FILE}")
            _grim_copy_one("${SRC_FILE}" "${DST_FILE}")
            add_custom_command(TARGET grim_copy_resources POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${SRC_FILE}" "${DST_FILE}"
                VERBATIM)
        endif()
    endforeach()

    # font (optional)
    if(EXISTS "${GRIM_FONT_SRC}")
        add_custom_command(TARGET grim_copy_resources POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${GRIM_FONT_SRC}" "${CMAKE_BINARY_DIR}/ui_font.ttf"
            VERBATIM)
    endif()

    # whisper model (optional)
    if(EXISTS "${WHISPER_MODEL_FILE}")
        add_custom_command(TARGET grim_copy_resources POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/models"
            COMMAND ${CMAKE_COMMAND} -E copy_if_different "${WHISPER_MODEL_FILE}" "${CMAKE_BINARY_DIR}/models/"
            VERBATIM)
    endif()
endif()

# ---------- attach to GRIM if present ----------
if(TARGET GRIM)
    add_dependencies(GRIM grim_copy_resources)
endif()

# ---------- install rules (safe if included multiple times) ----------
include(GNUInstallDirs)

foreach(FILE ${GRIM_RESOURCES})
    if(EXISTS "${CMAKE_SOURCE_DIR}/resources/${FILE}")
        install(FILES "${CMAKE_SOURCE_DIR}/resources/${FILE}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
    elseif(EXISTS "${CMAKE_BINARY_DIR}/${FILE}")
        install(FILES "${CMAKE_BINARY_DIR}/${FILE}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
    endif()
endforeach()

if(EXISTS "${GRIM_FONT_SRC}")
    install(FILES "${GRIM_FONT_SRC}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim)
endif()

if(EXISTS "${WHISPER_MODEL_FILE}")
    install(FILES "${WHISPER_MODEL_FILE}" DESTINATION ${CMAKE_INSTALL_DATADIR}/grim/models)
endif()
