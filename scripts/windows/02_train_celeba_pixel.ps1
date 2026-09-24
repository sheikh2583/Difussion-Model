# Stage 2: fresh, newly numbered CelebA pixel suite. Remaining options are forwarded.
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]]$RemainingArguments
)
$arguments = @("-Dataset", "celeba", "-Mode", "fresh")
$arguments += $RemainingArguments
& "$PSScriptRoot\train_all_datasets.ps1" @arguments
exit $LASTEXITCODE

