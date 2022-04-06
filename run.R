clean_thesis_folder <- function(folder = ".") {
  all_files <- list.files(folder, full.names = TRUE)
  to_delete <- c(".log", ".maf", ".mtc", ".bbl", ".blg", ".run.xml", ".synctex.gz", ".pdf")
  file.remove(all_files[grepl(pattern = paste0(to_delete, collapse = "|"), x = all_files)])
}


clean_thesis_folder()
  