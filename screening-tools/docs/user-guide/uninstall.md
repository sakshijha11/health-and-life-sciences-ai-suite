<!--hide_directive
```{eval-rst}
.. meta::
   :description: How to remove the Screening Tools sample application and its data.
   :keywords: uninstall, cleanup, remove
```
hide_directive-->

# Uninstall

The application installs nothing outside its own folder. It does not change
system settings, services or the registry.

1. Stop any running instance with **Ctrl+C** in its console.
2. Keep any results you need from `runs\`.
3. Delete the application folder. This removes the virtual environment
   (`venv\`), the models (`models\`), the OpenVINO™ model cache (`cache\`) and
   all run results (`runs\`):

   ```powershell
   cd ..
   Remove-Item -Recurse -Force .\screening-tools
   ```

To remove only part of it, from the application folder:

| Remove | Command | Effect |
| --- | --- | --- |
| Run results | `Remove-Item -Recurse -Force .\runs` | Results are gone; the application still works |
| Model cache | `Remove-Item -Recurse -Force .\cache` | The first run compiles the model again |
| Models | `Remove-Item .\models\* -Exclude README.md` | Run `python -m screening_tools setup` before the next run |
| Virtual environment | `Remove-Item -Recurse -Force .\venv` | Run `.\setup.ps1` before the next run |

`pip` may keep downloaded packages in its own cache. To clear it, run
`.\venv\Scripts\python.exe -m pip cache purge` before deleting `venv\`.
