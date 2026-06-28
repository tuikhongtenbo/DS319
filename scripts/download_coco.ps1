param(
    [string]$ImageRoot,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Resolve-Path (Join-Path $ScriptDir "..")

if ([string]::IsNullOrWhiteSpace($ImageRoot)) {
    $ImageRoot = Join-Path $RootDir "data\images"
}

$ImageRoot = [System.IO.Path]::GetFullPath($ImageRoot)
$ZipPath = Join-Path $ImageRoot "test2017.zip"
$ExtractedDir = Join-Path $ImageRoot "test2017"
$CocoDir = Join-Path $ImageRoot "COCO2017"
$Url = "http://images.cocodataset.org/zips/test2017.zip"

New-Item -ItemType Directory -Force -Path $ImageRoot | Out-Null

if ((Test-Path $CocoDir) -and -not $Force) {
    Write-Host "COCO2017 images already exist at $CocoDir"
    exit 0
}

if ($Force) {
    if (Test-Path $CocoDir) {
        Remove-Item -LiteralPath $CocoDir -Recurse -Force
    }
    if (Test-Path $ExtractedDir) {
        Remove-Item -LiteralPath $ExtractedDir -Recurse -Force
    }
}

if (-not (Test-Path $ZipPath)) {
    Write-Host "Downloading COCO test2017 images..."
    try {
        Start-BitsTransfer -Source $Url -Destination $ZipPath
    }
    catch {
        Write-Host "BITS download failed, falling back to Invoke-WebRequest..."
        Invoke-WebRequest -Uri $Url -OutFile $ZipPath
    }
}
else {
    Write-Host "Found existing archive at $ZipPath"
}

if (-not (Test-Path $CocoDir)) {
    Write-Host "Extracting archive..."
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $ImageRoot -Force

    if (Test-Path $ExtractedDir) {
        Move-Item -LiteralPath $ExtractedDir -Destination $CocoDir
    }
    elseif (-not (Test-Path $CocoDir)) {
        throw "Expected extracted folder was not found: $ExtractedDir"
    }
}

Write-Host "COCO2017 images ready at $CocoDir"
