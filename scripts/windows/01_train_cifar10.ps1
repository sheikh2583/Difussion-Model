# Stage 1: fresh, newly numbered CIFAR-10 suite. Remaining options are forwarded.
[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [object[]]$RemainingArguments
)
$arguments = @("-Dataset", "cifar10", "-CifarBackbone", "current", "-Mode", "fresh")
$arguments += $RemainingArguments
& "$PSScriptRoot\train_all_datasets.ps1" @arguments
exit $LASTEXITCODE

