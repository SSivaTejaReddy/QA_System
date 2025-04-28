def choose_language(supported_languages):
    print("\nLanguages for translation:")
    print(", ".join(supported_languages))

    while True:
        user_input = input("Enter the target language: ").strip()

        matches = [lang for lang in supported_languages if lang.lower() == user_input.lower()]
        if matches:
            return matches[0]
        else:
            print("Unsupported language. Please choose from the list.")