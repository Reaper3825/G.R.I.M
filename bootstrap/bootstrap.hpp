#pragma once
#include <string>
#include "../MMO/Core/HardwareInventory.hpp"
#include "../MMO/Core/ResourceSignal.hpp"
#include "../MMO/Core/ResourceCoordinator.hpp"
#include "../MMO/Core/ModelRegistry.hpp"
#include "../MMO/Core/ModelLoader.hpp"
#include "../MMO/Core/ProcessManager.hpp"
#include "../MMO/Core/Orchestrator.hpp"
#include "../memory/MemoryFacade.hpp"
#include "../MMO/Core/SessionContextManager.hpp"
#include "../MMO/Core/ToolRegistry.hpp"
#include "../MMO/Core/ActionPolicyRegistry.hpp"
#include "../MMO/UI/UISurfaceRegistry.hpp"

// Run startup checks and initialize GRIM environment
void runBootstrapChecks(int argc, char** argv);

// Stop MMO idle-tick background thread (call during shutdown)
void stopMMOIdleTick();

// Global resource layer — replaces old g_systemInfo
extern GRIM::MMO::HardwareInventory    g_hardwareInventory;
extern GRIM::MMO::ResourceSignal*      g_resourceSignal;
extern GRIM::MMO::ResourceCoordinator* g_resourceCoordinator;

// Global MMO orchestration layer
extern GRIM::MMO::ModelLoader*      g_modelLoader;
extern GRIM::MMO::ProcessManager*   g_processManager;
extern GRIM::MMO::Orchestrator*     g_orchestrator;
extern GRIM::MMO::MemoryFacade*     g_memoryFacade;
