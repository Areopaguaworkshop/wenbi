function Pandoc(doc)
    local function is_deletion(elem)
        return elem.classes and elem.classes:find('deletion')
    end

    local function is_addition(elem)
        return elem.classes and elem.classes:find('addition')
    end

    -- Track changes in text
    doc:walk {
        Str = function(elem)
            if is_deletion(elem) then
                elem.text = '{-' .. elem.text .. '-}'
            elseif is_addition(elem) then
                elem.text = '{+' .. elem.text .. '+}'
            end
        end,
        
        Note = function(elem)
            -- Preserve footnotes
            return elem
        end
    }
    
    return doc
end
