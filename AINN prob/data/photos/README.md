# Your own photos go here

The image branch was evaluated on simulated phone photos. The strongest check you can add is real
photos taken with your own phone:

1. `python scripts/eval_user_photos.py --export-pages` writes the committed paper's pages to
   `data/photos/print_me/` (PNG files plus `print_me.pdf`).
2. Print one or two pages. Photograph them any way a leaker would: at an angle, in bad light, only
   part of the page, a screenshot of the photo, a copy forwarded through WhatsApp.
3. Also photograph things that are **not** the paper (your notes, a textbook, a different printed
   paper). These are the checks against false alarms.
4. Copy all the photos into this folder and run `python scripts/eval_user_photos.py`.

A verified score of 0.5 or more counts as a match. The script also prints which page it thinks the
photo shows.
