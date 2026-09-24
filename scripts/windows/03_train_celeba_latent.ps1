# Stage 3: fresh, newly numbered CelebA latent suite. Remaining options are forwarded.
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]]$RemainingArguments
)
$arguments = @("-Mode", "fresh")
$arguments += $RemainingArguments
& "$PSScriptRoot\train_celeba_latent.ps1" @arguments
exit $LASTEXITCODE

