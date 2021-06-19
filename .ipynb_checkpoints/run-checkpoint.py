{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "43808ae2-93a9-4320-b1c9-6833ae55575b",
   "metadata": {},
   "outputs": [],
   "source": [
    "from flask import Flask\n",
    "app = Flask(__name__)\n",
    "\n",
    "@app.route(/)\n",
    "def home():\n",
    "    return 'hello'\n",
    "\n",
    "if __name__=='__main__':\n",
    "    app.run(debug=True)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.9"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
