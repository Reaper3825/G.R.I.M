"""CMake plugin publication tests; never configure or build the GRIM application."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "plugins/deploy_plugin.cmake"


class PluginDeploymentTests(unittest.TestCase):
    def setUp(self):
        self.base = ROOT / ".codex-build/plugin-deployment-tests"
        self.base.mkdir(parents=True, exist_ok=True)
        self.folder = Path(tempfile.mkdtemp(prefix="publication ", dir=self.base))

    def tearDown(self):
        # Recursive cleanup is restricted to this test's verified workspace folder.
        self.assertTrue(self.folder.resolve().is_relative_to(self.base.resolve()))
        shutil.rmtree(self.folder)

    def run_command(self, *args, success=True):
        result = subprocess.run(args, capture_output=True, text=True)
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def publish(self, source, destination, *options, success=True):
        return self.run_command("cmake", f"-DPLUGIN_FILE={source}",
                                f"-DDEPLOY_DIR={destination}", *options, "-P", str(SCRIPT), success=success)

    def test_publish_repair_and_unchanged_timestamp(self):
        source = self.folder / "fixture.dll"
        source.write_bytes(b"completed plugin")
        destination = self.folder / "live plugins"
        self.publish(source, destination)
        deployed = destination / source.stem / source.name
        self.assertEqual(deployed.read_bytes(), source.read_bytes())
        # An unchanged build must leave the deployed binary untouched.
        os.utime(deployed, (946684800, 946684800))
        timestamp = deployed.stat().st_mtime_ns
        self.publish(source, destination)
        self.assertEqual(deployed.stat().st_mtime_ns, timestamp)
        deployed.write_bytes(b"stale plugin")
        self.publish(source, destination)
        self.assertEqual(deployed.read_bytes(), source.read_bytes())
        deployed.unlink()
        self.publish(source, destination)
        self.assertEqual(deployed.read_bytes(), source.read_bytes())
        self.assertEqual(list(destination.rglob("*.pending")), [])

    def test_missing_output_preserves_live_plugin(self):
        destination = self.folder / "live"
        deployed = destination / "fixture/fixture.dll"
        deployed.parent.mkdir(parents=True)
        deployed.write_bytes(b"working plugin")
        self.publish(self.folder / "missing/fixture.dll", destination, success=False)
        self.assertEqual(deployed.read_bytes(), b"working plugin")
        self.assertEqual(list(deployed.parent.iterdir()), [deployed])

    def test_output_already_in_deployment_folder(self):
        source = self.folder / "fixture/fixture.dll"
        source.parent.mkdir()
        source.write_bytes(b"completed plugin")
        self.publish(source, self.folder)
        self.assertEqual(source.read_bytes(), b"completed plugin")
        self.assertEqual(list(self.folder.iterdir()), [source.parent])
        self.assertFalse((source.parent / ".deployment").exists())

    def test_publish_manifest_and_private_dependency(self):
        binary = self.folder / "fixture.dll"
        manifest = self.folder / "fixture.tools.json"
        dependency = self.folder / "helper.dll"
        destination = self.folder / "live"
        bundle = destination / "fixture"
        binary.write_bytes(b"completed plugin")
        manifest.write_text("{}")
        dependency.write_bytes(b"runtime library")
        options = (f"-DPLUGIN_MANIFEST={manifest}", f"-DPLUGIN_RUNTIME_DLLS={dependency}")
        self.publish(binary, destination, *options)
        self.assertEqual((bundle / binary.name).read_bytes(), binary.read_bytes())
        self.assertEqual((bundle / manifest.name).read_bytes(), manifest.read_bytes())
        deployed_dependency = bundle / "dependencies" / dependency.name
        self.assertEqual(deployed_dependency.read_bytes(), dependency.read_bytes())

        dependency.write_bytes(b"updated runtime library")
        self.publish(binary, destination, *options)
        self.assertEqual(deployed_dependency.read_bytes(), dependency.read_bytes())

        manifest.unlink()
        self.publish(binary, destination, *options, success=False)
        self.assertEqual((bundle / manifest.name).read_text(), "{}")

        manifest.write_text("{}")
        self.publish(binary, destination, f"-DPLUGIN_MANIFEST={manifest}")
        self.assertFalse(deployed_dependency.exists())

    def test_host_platforms_include_complete_standalone_catalog(self):
        # Evaluate the actual CMake platform conditions, intercepting target
        # registration so this never configures the host or its dependencies.
        host_plugins = (ROOT / "CMakeLists.txt").read_text().split(
            "# Hot-Reloadable Plugins", 1)[1]
        expected = {p.parent.name for p in (ROOT / "plugins").glob("*/CMakeLists.txt")}
        for platform in ("windows", "macos"):
            with self.subTest(platform=platform):
                output = self.folder / f"{platform}-plugins.txt"
                script = self.folder / f"{platform}.cmake"
                script.write_text(
                    'cmake_minimum_required(VERSION 3.22)\n'
                    f'set(WIN32 {"TRUE" if platform == "windows" else "FALSE"})\n'
                    f'set(APPLE {"TRUE" if platform == "macos" else "FALSE"})\n'
                    'function(project)\nendfunction()\n'
                    'function(target_link_libraries)\nendfunction()\n'
                    'function(add_subdirectory path)\n'
                    '  if(path STREQUAL "plugins")\n'
                    f'    include("{(ROOT / "plugins/CMakeLists.txt").as_posix()}")\n'
                    '  else()\n'
                    '    get_filename_component(name "${path}" NAME)\n'
                    f'    file(APPEND "{output.as_posix()}" "${{name}}\\n")\n'
                    '  endif()\nendfunction()\n' + host_plugins)
                self.run_command("cmake", "-P", str(script))
                actual = output.read_text().splitlines()
                self.assertEqual(set(actual), expected)
                self.assertEqual(len(actual), len(expected))

    def configure_fixture(self, host, deployment=None):
        source = self.folder / "project"
        source.mkdir()
        # A synthetic host target tests dependency wiring without any executable,
        # model, runtime service, third-party package, or actual GRIM CMake project.
        host_target = "add_custom_target(GRIM)\n" if host else ""
        (source / "CMakeLists.txt").write_text(
            'cmake_minimum_required(VERSION 3.22)\nproject(DeploymentFixture LANGUAGES CXX)\n'
            + host_target + f'include("{(ROOT / "plugins/plugin_build.cmake").as_posix()}")\n'
            + 'add_library(fixture_plugin SHARED fixture.cpp)\ngrim_configure_plugin(fixture_plugin)\n'
            + 'file(WRITE "${CMAKE_BINARY_DIR}/deployment-enabled.txt" "${GRIM_DEPLOY_PLUGINS}")\n')
        (source / "fixture.cpp").write_text('extern "C" int deployment_fixture() { return 1; }\n')
        build = self.folder / "build"
        destination = self.folder / "live"
        args = ["cmake", "-S", str(source), "-B", str(build), f"-DGRIM_PLUGIN_DEPLOY_DIR={destination}"]
        if os.name == "nt":
            args += ["-G", "Visual Studio 17 2022", "-A", "x64"]
        if deployment is not None:
            args += [f"-DGRIM_DEPLOY_PLUGINS={deployment}"]
        self.run_command(*args)
        return build, destination

    def test_host_build_publishes_and_repairs_without_relinking(self):
        build, destination = self.configure_fixture(host=True)
        self.assertEqual((build / "deployment-enabled.txt").read_text(), "ON")
        command = ["cmake", "--build", str(build), "--config", "Release", "--target", "GRIM", "--parallel", "2"]
        self.run_command(*command)
        libraries = [p for p in destination.glob("*/*") if p.suffix in (".dll", ".so", ".dylib")]
        self.assertEqual(len(libraries), 1)
        deployed = libraries[0]
        output = build / "plugins" / deployed.name
        timestamp = output.stat().st_mtime_ns
        expected = hashlib.sha256(output.read_bytes()).digest()
        deployed.write_bytes(b"stale deployment")
        self.run_command(*command)
        self.assertEqual(hashlib.sha256(deployed.read_bytes()).digest(), expected)
        self.assertEqual(output.stat().st_mtime_ns, timestamp)

    def test_standalone_defaults_to_no_deployment(self):
        build, destination = self.configure_fixture(host=False)
        self.assertEqual((build / "deployment-enabled.txt").read_text(), "OFF")
        self.run_command("cmake", "--build", str(build), "--config", "Release", "--parallel", "2")
        self.assertFalse(destination.exists())

    def test_standalone_can_opt_in(self):
        build, destination = self.configure_fixture(host=False, deployment="ON")
        self.run_command("cmake", "--build", str(build), "--config", "Release", "--target", "deploy_fixture_plugin", "--parallel", "2")
        self.assertTrue(any(destination.iterdir()))

    def test_host_can_opt_out(self):
        build, destination = self.configure_fixture(host=True, deployment="OFF")
        self.assertEqual((build / "deployment-enabled.txt").read_text(), "OFF")
        self.run_command("cmake", "--build", str(build), "--config", "Release", "--parallel", "2")
        self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main()
