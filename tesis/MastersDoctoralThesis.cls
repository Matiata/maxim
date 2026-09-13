%% MastersDoctoralThesis.cls
%% Reconstruccion compatible para la plantilla recibida por FaMAF/UNC.
%% No es una copia de la clase original de LaTeXTemplates.
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{MastersDoctoralThesis}[2026/09/09 v1.0 Compatibility thesis class]

\RequirePackage{kvoptions}
\SetupKeyvalOptions{family=MDT,prefix=MDT@}

\DeclareBoolOption[false]{nohyperref}
\DeclareBoolOption[false]{headsepline}
\DeclareBoolOption[false]{chapterinoneline}
\DeclareBoolOption[false]{consistentlayout}
\DeclareBoolOption[false]{nolistspacing}
\DeclareBoolOption[false]{liststotoc}
\DeclareBoolOption[false]{toctotoc}
\DeclareBoolOption[false]{parskip}

\newcommand{\MDT@spacing}{1}
\DeclareVoidOption{singlespacing}{\renewcommand{\MDT@spacing}{1}}
\DeclareVoidOption{onehalfspacing}{\renewcommand{\MDT@spacing}{1.5}}
\DeclareVoidOption{doublespacing}{\renewcommand{\MDT@spacing}{2}}

% Idiomas conocidos por babel. El ultimo indicado queda como idioma principal.
\DeclareVoidOption{english}{\PassOptionsToPackage{english}{babel}}
\DeclareVoidOption{spanish}{\PassOptionsToPackage{spanish}{babel}}
\DeclareVoidOption{ngerman}{\PassOptionsToPackage{ngerman}{babel}}
\DeclareVoidOption{french}{\PassOptionsToPackage{french}{babel}}

% Las opciones normales de book (11pt, oneside, draft, etc.) se reenvian.
\DeclareDefaultOption{\PassOptionsToClass{\CurrentOption}{book}}
\ProcessKeyvalOptions*
\LoadClass{book}

% `provide=*` permite compilar aun en instalaciones mínimas de TeX Live que
% no traen todos los archivos .ldf; Overleaf sí incluye soporte español.
\PassOptionsToPackage{provide=*}{babel}
\RequirePackage{babel}
\RequirePackage{graphicx}
\RequirePackage{geometry}
\RequirePackage{setspace}
\RequirePackage{fancyhdr}
\RequirePackage{booktabs}
\RequirePackage{tabularx}
\RequirePackage{array}
\RequirePackage{enumitem}
\RequirePackage{etoolbox}
\RequirePackage{xparse}

\ifMDT@parskip
  \RequirePackage{parskip}
\fi

\ifMDT@nohyperref\else
  \RequirePackage{hyperref}
  \hypersetup{
    colorlinks=true,
    linkcolor=blue,
    citecolor=blue,
    urlcolor=blue,
    bookmarksnumbered=true,
    bookmarksopen=true
  }
\fi

\AtBeginDocument{\setstretch{\MDT@spacing}}

\ifMDT@nolistspacing
  \setlist{nosep}
\fi

% Datos de la tesis y alias usados por main.tex.
\providecommand{\ttitle}{}
\providecommand{\authorname}{}
\providecommand{\supname}{}
\providecommand{\examname}{}
\providecommand{\degreename}{}
\providecommand{\addressname}{}
\providecommand{\subjectname}{}
\providecommand{\keywordnames}{}
\providecommand{\univname}{}
\providecommand{\deptname}{}
\providecommand{\groupname}{}
\providecommand{\facname}{}

\newcommand{\thesistitle}[1]{\gdef\ttitle{#1}}
\newcommand{\supervisor}[1]{\gdef\supname{#1}}
\newcommand{\examiner}[1]{\gdef\examname{#1}}
\newcommand{\degree}[1]{\gdef\degreename{#1}}
\newcommand{\addresses}[1]{\gdef\addressname{#1}}
\newcommand{\subject}[1]{\gdef\subjectname{#1}}
\newcommand{\keywords}[1]{\gdef\keywordnames{#1}}
\newcommand{\university}[1]{\gdef\univname{#1}}
\newcommand{\department}[1]{\gdef\deptname{#1}}
\newcommand{\group}[1]{\gdef\groupname{#1}}
\newcommand{\faculty}[1]{\gdef\facname{#1}}

% Conserva \author de la clase book y mantiene sincronizado \authorname.
\let\MDT@originalauthor\author
\renewcommand{\author}[1]{%
  \MDT@originalauthor{#1}%
  \gdef\authorname{#1}%
}

\newcommand{\HRule}{\rule{\linewidth}{0.5mm}}
\newcommand{\addchaptertocentry}[1]{\addcontentsline{toc}{chapter}{#1}}

% Encabezados para el cuerpo de la tesis.
\newcommand{\MDT@setheaders}{%
  \fancyhf{}%
  \if@twoside
    \fancyhead[LE]{\thepage}%
    \fancyhead[RO]{\thepage}%
    \fancyhead[RE]{\nouppercase{\leftmark}}%
    \fancyhead[LO]{\nouppercase{\rightmark}}%
  \else
    \fancyhead[L]{\nouppercase{\rightmark}}%
    \fancyhead[R]{\thepage}%
  \fi
  \ifMDT@headsepline
    \renewcommand{\headrulewidth}{0.4pt}%
  \else
    \renewcommand{\headrulewidth}{0pt}%
  \fi
  \renewcommand{\footrulewidth}{0pt}%
}
\pagestyle{fancy}
\MDT@setheaders
\fancypagestyle{thesis}{%
  \MDT@setheaders
}

% Paginas preliminares.
\providecommand{\abstractname}{Abstract}
\NewDocumentEnvironment{abstract}{}{%
  \cleardoublepage
  \thispagestyle{plain}
  \null\vfill
  \begin{center}\bfseries\Large\abstractname\end{center}
  \begin{quote}
}{%
  \end{quote}
  \vfill\null
  \cleardoublepage
}

\NewDocumentEnvironment{acknowledgements}{}{%
  \cleardoublepage
  \chapter*{Agradecimientos}
}{\cleardoublepage}

\NewDocumentEnvironment{abbreviations}{m}{%
  \cleardoublepage
  \chapter*{Abreviaturas}
  \begin{longtable}{#1}
}{\end{longtable}\cleardoublepage}

\NewDocumentEnvironment{constants}{m}{%
  \cleardoublepage
  \chapter*{Constantes físicas}
  \begin{longtable}{#1}
}{\end{longtable}\cleardoublepage}

\NewDocumentEnvironment{symbols}{m}{%
  \cleardoublepage
  \chapter*{Símbolos}
  \begin{longtable}{#1}
}{\end{longtable}\cleardoublepage}

\RequirePackage{longtable}

\newcommand{\dedicatory}[1]{%
  \cleardoublepage
  \thispagestyle{plain}
  \null\vfill
  \begin{flushright}\itshape #1\end{flushright}
  \vfill\null
  \cleardoublepage
}

% Inclusión opcional de listas en el índice.
\ifMDT@liststotoc
  \pretocmd{\listoffigures}{\addchaptertocentry{\listfigurename}}{}{}
  \pretocmd{\listoftables}{\addchaptertocentry{\listtablename}}{}{}
\fi
\ifMDT@toctotoc
  \pretocmd{\tableofcontents}{\addchaptertocentry{\contentsname}}{}{}
\fi

% Opción de capítulo compacto, conservada por compatibilidad.
\ifMDT@chapterinoneline
  \RequirePackage{titlesec}
  \titleformat{\chapter}[hang]
    {\normalfont\huge\bfseries}{\thechapter.}{0.75em}{}
\fi

\endinput
