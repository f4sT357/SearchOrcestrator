param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Start', 'Stop')]
    [string]$Action
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$composeDirectory = Join-Path $projectRoot 'firecrawl-selfhost'
$composeFile = Join-Path $composeDirectory 'docker-compose.yaml'
$dockerPath = $null

function Find-DockerCli {
    $command = Get-Command docker -ErrorAction SilentlyContinue
    if ($command -and $command.Source -and (Test-Path -LiteralPath $command.Source)) {
        return $command.Source
    }

    $candidates = @(
        (Join-Path $env:ProgramFiles 'Docker\Docker\resources\bin\docker.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe')
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }
    return $null
}

function New-RandomHex([int]$ByteCount) {
    $bytes = New-Object byte[] $ByteCount
    $random = [Security.Cryptography.RandomNumberGenerator]::Create()
    try {
        $random.GetBytes($bytes)
    }
    finally {
        $random.Dispose()
    }
    return -join ($bytes | ForEach-Object { $_.ToString('x2') })
}

function Ensure-DockerEngine {
    $script:dockerPath = Find-DockerCli
    if (-not $script:dockerPath) {
        throw 'Docker CLI が見つかりません。Docker Desktop をインストールしてください。'
    }

    $dockerDirectory = Split-Path -Parent $script:dockerPath
    $env:PATH = $dockerDirectory + [IO.Path]::PathSeparator + $env:PATH

    $engineVersion = & $script:dockerPath info '--format' '{{.ServerVersion}}' 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Docker Engine: $engineVersion"
        return
    }

    $desktopCandidates = @(
        (Join-Path $env:ProgramFiles 'Docker\Docker\Docker Desktop.exe'),
        (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe')
    )
    $desktopPath = $desktopCandidates | Where-Object { $_ -and (Test-Path -LiteralPath $_) } | Select-Object -First 1
    if (-not $desktopPath) {
        throw 'Docker Engine に接続できず、Docker Desktop も見つかりません。Docker Desktop を起動してください。'
    }

    Write-Host 'Docker Desktop を起動しています。初回起動は数分かかる場合があります。'
    Start-Process -FilePath $desktopPath
    $deadline = (Get-Date).AddMinutes(4)
    do {
        Start-Sleep -Seconds 5
        $engineVersion = & $script:dockerPath info '--format' '{{.ServerVersion}}' 2>$null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Docker Engine: $engineVersion"
            return
        }
    } while ((Get-Date) -lt $deadline)

    throw 'Docker Desktop は起動しましたが、Docker Engine が4分以内に応答しませんでした。Docker Desktop の状態を確認してください。'
}

function Initialize-FirecrawlCheckout {
    if (-not (Test-Path -LiteralPath $composeFile)) {
        $git = Get-Command git -ErrorAction SilentlyContinue
        if (-not $git) {
            throw 'Firecrawl が見つからず Git も利用できません。Git をインストールしてから再実行してください。'
        }

        Write-Host 'Firecrawl v2.11.162 を取得しています（初回のみ）。'
        & $git.Source clone --depth 1 --branch v2.11.162 https://github.com/firecrawl/firecrawl.git $composeDirectory
        if ($LASTEXITCODE -ne 0) {
            throw 'Firecrawl の取得に失敗しました。ネットワークと Git の状態を確認してください。'
        }
    }

    if (-not (Test-Path -LiteralPath $composeFile)) {
        throw "Docker Compose ファイルが見つかりません: $composeFile"
    }

    $envFile = Join-Path $composeDirectory '.env'
    if (-not (Test-Path -LiteralPath $envFile)) {
        $dbPassword = New-RandomHex -ByteCount 32
        $adminKey = New-RandomHex -ByteCount 32
        $settings = @(
            'PORT=127.0.0.1:3002',
            'USE_DB_AUTHENTICATION=false',
            'POSTGRES_USER=postgres',
            "POSTGRES_PASSWORD=$dbPassword",
            'POSTGRES_DB=postgres',
            "BULL_AUTH_KEY=$adminKey",
            ''
        ) -join "`n"
        [IO.File]::WriteAllText($envFile, $settings, [Text.UTF8Encoding]::new($false))
        Write-Host 'ローカル接続用の .env を作成しました。'
    }

    $envContents = [IO.File]::ReadAllText($envFile)
    $adminKeyLine = [Text.RegularExpressions.Regex]::Match($envContents, '(?m)^BULL_AUTH_KEY=.*$')
    if (-not $adminKeyLine.Success -or $adminKeyLine.Value -eq 'BULL_AUTH_KEY=') {
        $adminKey = New-RandomHex -ByteCount 32
        if ($adminKeyLine.Success) {
            $envContents = [Text.RegularExpressions.Regex]::Replace($envContents, '(?m)^BULL_AUTH_KEY=.*$', "BULL_AUTH_KEY=$adminKey", 1)
            [IO.File]::WriteAllText($envFile, $envContents, [Text.UTF8Encoding]::new($false))
        }
        else {
            $separator = if ($envContents.EndsWith("`n") -or $envContents.EndsWith("`r")) { '' } else { "`n" }
            [IO.File]::AppendAllText($envFile, "${separator}BULL_AUTH_KEY=$adminKey`n", [Text.UTF8Encoding]::new($false))
        }
        Write-Host 'Bull Queue の管理キーを .env に追加しました。'
    }
}

function Invoke-FirecrawlCompose([string[]]$Arguments) {
    Push-Location $composeDirectory
    try {
        & $script:dockerPath compose @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "docker compose $($Arguments -join ' ') に失敗しました（終了コード $LASTEXITCODE）。"
        }
    }
    finally {
        Pop-Location
    }
}

try {
    if ($Action -eq 'Start') {
        Initialize-FirecrawlCheckout
        Ensure-DockerEngine
        Write-Host 'Firecrawl を起動しています。初回はイメージのビルドに時間がかかります。'
        Invoke-FirecrawlCompose -Arguments @('up', '--build', '-d')

        $deadline = (Get-Date).AddMinutes(4)
        $ready = $false
        do {
            try {
                $health = Invoke-RestMethod -Uri 'http://127.0.0.1:3002/v0/health/readiness' -TimeoutSec 5
                if ($health.status -eq 'ok') {
                    $ready = $true
                    break
                }
            }
            catch {
                Start-Sleep -Seconds 3
            }
        } while ((Get-Date) -lt $deadline)

        if (-not $ready) {
            throw 'コンテナは起動しましたが API が4分以内に ready になりませんでした。docker compose logs api でログを確認してください。'
        }
        Write-Host 'Firecrawl は起動済みです: http://localhost:3002'
    }
    else {
        if (-not (Test-Path -LiteralPath $composeFile)) {
            Write-Host 'Firecrawl の Compose 構成がありません。停止するコンテナはありません。'
            exit 0
        }
        $script:dockerPath = Find-DockerCli
        if (-not $script:dockerPath) {
            throw 'Docker CLI が見つかりません。Docker Desktop をインストールしてください。'
        }
        $dockerDirectory = Split-Path -Parent $script:dockerPath
        $env:PATH = $dockerDirectory + [IO.Path]::PathSeparator + $env:PATH
        $engineVersion = & $script:dockerPath info '--format' '{{.ServerVersion}}' 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host 'Docker Engine が停止中です。Firecrawl も停止しています。'
            exit 0
        }
        Invoke-FirecrawlCompose -Arguments @('down')
        Write-Host 'Firecrawl のコンテナを停止しました。Docker Desktop は起動したままです。'
    }
}
catch {
    Write-Host "エラー: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
