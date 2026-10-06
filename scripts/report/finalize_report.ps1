# finalize_report.ps1 - opens the raw report in word, builds the contents lists, saves the final docx and a pdf copy
param(
  [string]$Raw = "C:\Users\91738\edge-waste\docs\report\BITE497J_Project_I_Report_raw.docx",
  [string]$Out = "C:\Users\91738\edge-waste\docs\report\BITE497J_Project_I_Report.docx",
  [string]$Pdf = "C:\Users\91738\edge-waste\docs\report\BITE497J_Project_I_Report.pdf"
)
$w = New-Object -ComObject Word.Application
$w.Visible = $false
$w.DisplayAlerts = 0
try {
  $doc = $w.Documents.Open($Raw, $false, $false)
  $doc.Repaginate()
  for ($pass = 0; $pass -lt 2; $pass++) {
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    $doc.Repaginate()
  }
  "tables of contents: " + $doc.TablesOfContents.Count
  $doc.SaveAs2($Out, 16)
  $doc.ExportAsFixedFormat($Pdf, 17)
  "pages: " + $doc.ComputeStatistics(2)
  $doc.Close($false)
} finally {
  $w.Quit()
}
