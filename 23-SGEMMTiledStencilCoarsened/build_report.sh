#!/bin/sh
mkdir -p build_report_tmp
cat description.markdown > build_report_tmp/description.md
echo "" >> build_report_tmp/description.md
echo "# Questions" >>  build_report_tmp/description.md
cat questions.markdown >>  build_report_tmp/description.md
cat ../support/description/common.markdown >> build_report_tmp/description.md
cp ../support/description/template/* build_report_tmp/
cd build_report_tmp
pandoc -s -N --template=template.tex description.md -o description.tex
pdflatex description.tex -o description.pdf
cd ..
cp build_report_tmp/description.pdf .
rm -fr build_report_tmp

