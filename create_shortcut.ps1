$DesktopPath = [Environment]::GetFolderPath("Desktop")
$ShortcutPath = Join-Path $DesktopPath "Customer Churn Predictor.lnk"
$Target = "C:\Users\Admin\Desktop\minor project\launch.bat"

echo "Creating desktop shortcut at: $ShortcutPath"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($ShortcutPath)
$Shortcut.TargetPath = $Target
$Shortcut.WorkingDirectory = "C:\Users\Admin\Desktop\minor project"
$Shortcut.Description = "Launch the Customer Churn Prediction Web Application"
$Shortcut.IconLocation = "shell32.dll,220"
$Shortcut.Save()

echo "Shortcut successfully created!"
